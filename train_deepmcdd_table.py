import os, argparse
import torch
import numpy as np
import itertools
import models
import sklearn
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix
from dataloader_table import get_table_data
# from dataloader_table import get_table_data_with_iden
from dataloader_table import get_table_data_with_iden_separate_sets
from utils import compute_confscores, compute_metrics, print_ood_results, print_ood_results_total
#from sklearn.semi_supervised import SelfTrainingClassifier # for Semi-Supervised learning
#from datetime import date
#from datetime import timedelta

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='gas | shuttle | drive | mnist')
parser.add_argument('--net_type', required=True, help='mlp')
# parser.add_argument('--num_classes', required=True,type=int, help='number of classes including in dist and ood')
# parser.add_argument('--num_features', required=True,type=int, help='number of features in dataset')
parser.add_argument('--conf', type=float, default=0.9, help='confidence for marking id')
parser.add_argument('--focal_gamma', type=float, default= 1, help=' gamma = 0 means CE , for focal use gamma 1 or so')
parser.add_argument('--model_path', default=0, help='path to saved model')
parser.add_argument('--datadir', default='./table_data/', help='path to dataset')
parser.add_argument('--outdir', default='./output/', help='folder to output results')
parser.add_argument('--oodclass_idx', type=int, default=0, help='index of the OOD class')
parser.add_argument('--batch_size', type=int, default=200, help='batch size for data loader')
parser.add_argument('--latent_size', type=int, default=128, help='dimension size for latent representation') 
parser.add_argument('--num_layers', type=int, default=3, help='the number of hidden layers in MLP')
parser.add_argument('--num_folds', type=int, default=5, help='the number of cross-validation folds')
parser.add_argument('--num_epochs', type=int, default=10, help='the number of epochs for training sc-layers')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate of Adam optimizer')
parser.add_argument('--reg_pull', type=float, default=1.0, help='regularization coefficient for min dist')
parser.add_argument('--reg_focal1', type=float, default=1.0, help='regularization coefficient for dists')
parser.add_argument('--reg_focal2', type=float, default=1.0, help='regularization coefficient for scores')
parser.add_argument('--reg_push', type=float, default=0.01, help='regularization coefficient for scores based mean loss')
parser.add_argument('--gpu', type=int, default=-1, help='gpu index')

args = parser.parse_args()
print(args)

# Get today's date
#today = date.today()
#rundate = datetime.strftime(today,"%Y%m%d")
#print(rundate)

def main():
    outdir = os.path.join(args.outdir, args.net_type + '_' + args.dataset)

    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # torch.cpu.set_device(args.gpu)

    best_idacc_list, best_oodacc_list = [], []
    for fold_idx in range(args.num_folds):
        
        train_loader, test_id_loader, test_ood_loader, num_features1, num_classes1 = get_table_data_with_iden_separate_sets(args.batch_size, args.datadir, args.dataset, args.oodclass_idx, fold_idx)
        print(f' shape of complete train data: {len(train_loader.dataset)}, of validation : {len(test_id_loader.dataset)} of ood:{ len(test_ood_loader.dataset)}')
        if args.dataset == 'gas':
            num_classes, num_features = 6, 128
        elif args.dataset == 'drive':
            num_classes, num_features = 11, 48
        elif args.dataset == 'shuttle':
            num_classes, num_features = 7, 9
        elif args.dataset == 'mnist':
            num_classes, num_features = 10, 784
        else: 
          num_classes, num_features = num_classes1, num_features1 
#           num_classes, num_features = args.num_classes, args.num_features
        print(f' number of classes : {num_classes}, number of features: {num_features}, no of id classes: {num_classes-1}' )
        
        #model_mcdd = models.MLP_DeepMCDD(num_features, args.num_layers*[args.latent_size], num_classes=num_classes-1)
        if args.model_path == 0:
           model = models.MLP_DeepMCDD(num_features, args.num_layers*[args.latent_size], num_classes=num_classes-1)
        else:
            model = torch.load(args.model_path) 
        # ## semi-supervised-------
        # model = SelfTrainingClassifier(base_estimator=model_mcdd, # An estimator object implementing fit and predict_proba.
        #                                      threshold=0.7, # default=0.75, The decision threshold for use with criterion='threshold'. Should be in [0, 1).
        #                                      criterion='threshold', # {‘threshold’, ‘k_best’}, default=’threshold’, The selection criterion used to select which labels to add to the training set. If 'threshold', pseudo-labels with prediction probabilities above threshold are added to the dataset. If 'k_best', the k_best pseudo-labels with highest prediction probabilities are added to the dataset.
        #                                      #k_best=50, # default=10, The amount of samples to add in each iteration. Only used when criterion='k_best'.
        #                                      max_iter=100, # default=10, Maximum number of iterations allowed. Should be greater than or equal to 0. If it is None, the classifier will continue to predict labels until no new pseudo-labels are added, or all unlabeled samples have been labeled.
        #                                      verbose=True # default=False, Verbosity prints some information after each iteration
        #                                     )
        model.cpu()
        #class MLP_DeepMCDD(nn.Module):
        #def __init__(self, input_size, hidden_sizes, num_classes)
        # latent_size = dimension size for latent representation, default = 128
        # num_layers = the number of hidden layers in MLP, default = 3
        # ce_loss = torch.nn.CrossEntropyLoss()
        weights = torch.ones(num_classes-1).cpu()
        def ce_loss(predicted, labels, weights=None):
            """
            Compute cross-entropy loss with optional class weights.
    
            Args:
                predicted (torch.Tensor): Predicted logits from the model.
                labels (torch.Tensor): True labels.
                weights (torch.Tensor, optional): Weights for each class. Default is None.
    
            Returns:
                torch.Tensor: Cross-entropy loss.
            """
            # Apply weights if provided
            if weights is not None:
                cross_entropy = torch.nn.functional.cross_entropy(predicted, labels, weight=weights)
            else:
                cross_entropy = torch.nn.functional.cross_entropy(predicted, labels)
    
            return cross_entropy
            
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
   
        #idacc_list1, idacc_list2, idacc_list3, idacc_list4, idacc_list5, idacc_list6, idacc_list7, idacc_list8, idacc_list9 = [],[],[],[],[],[],[],[],[]
        #--------------------------TRAIN DATA--------------------------------------
        idacc_list, oodacc_list = [], []
        #out_feat = []
        total_step = len(train_loader)
        for epoch in range(args.num_epochs):
            out_feat_train = []
            labels_list = []
            model.train()
            total_loss = 0.0
            total_data = 0
            for i, (data, labels) in enumerate(train_loader):
                
                data, labels = data.cpu(), labels.cpu()

                dists, out, out_big = model(data)
                scores = (- dists + model.alphas)
                # scores = (- dists + model.alphas)/model.alphas
                # print(scores.shape)
                # print(scores)
                conf = torch.exp(-torch.min(dists, dim=1).values)
                _, predicted = torch.max(scores, 1)
                # print(dists)
                # print(dists.shape)
                # dists_norm = dists / dists.sum(axis=1, keepdims=True)
                label_mask = torch.zeros(labels.size(0), model.num_classes).cpu().scatter_(1, labels.unsqueeze(dim=1), 1)
                pull_loss = torch.mean(torch.sum(torch.mul(label_mask, dists), dim=1))
                push_loss_scores = torch.mean(torch.sum(torch.mul(1-label_mask, torch.exp(scores)), dim=1))
                #------class balanced cross entropy loss ---------
                classes, count = labels.unique(return_counts=True, sorted=True)
                class_count_dict = dict(zip(classes.tolist(), count.tolist()))
                # print(classes, count)
                betaf = (labels.shape[0] -1)/labels.shape[0]
                # alphaf = (1-betaf)/(1 - betaf**count)
                for i in classes.tolist():
                     weights[i] = (1-betaf)/(1 - betaf**class_count_dict[i])
                # Normalize each row
                # scores_norm = scores / scores.sum(axis=1, keepdims=True)  
                # distance is the distance of data point from the class --- less the distance more is the probabbility
                # so i will inverse the distance and hten normalise to send in focal loss
                # Take the inverse of each element
                dists = 1 / (dists + torch.exp(torch.tensor(10**(-12.0))))

                # Normalize each row
                # norm_dists= dists / dists.sum(axis=1, keepdims=True)
                norm_dists= dists
                # # Verify that each row has a sum of 1
                # row_sums = norm_dists.sum(axis=1)
                # print(row_sums)
                # push_loss = ce_loss(scores, labels, weights)
                push_loss1 = ce_loss(norm_dists, labels, weights)
                push_loss2 = ce_loss(scores, labels, weights)
                # push_loss = ce_loss(scores, labels) --- for what they have doen in deep mcdd -- use gamma 0 as well
                pt1 = torch.exp(-push_loss1)
                focal_loss1 = ((1-pt1)**args.focal_gamma * push_loss1).mean()

                pt2 = torch.exp(-push_loss2)
                focal_loss2 = ((1-pt2)**args.focal_gamma * push_loss2).mean()

                loss = args.reg_pull * pull_loss + args.reg_focal1*focal_loss1 + args.reg_focal2*focal_loss2 + args.reg_push*push_loss_scores
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() 
                for j in range(len(out_big)):
                    #labels_list.append(labels[j].squeeze().tolist())
                    out_feat_train.append(out_big[j][0].squeeze().tolist()+ [labels[j].squeeze().tolist()]+ [predicted[j].squeeze().tolist()] +scores[j,:].squeeze().tolist()+ [conf[j].squeeze().tolist()] )
            print(f'loss for epoch {epoch} is : {loss}')
            # print(f"centre from train: {model.centers}")
            # print(f"for train radii:{radii1}, \n alpha : {param}")

            model.eval()
            with torch.no_grad():

                correct, total = 0, 0
                #correct1, total1 = 0, 0
                out_feat_test = []

                
                # dataloader_iterator = iter(test_id_loader)
                
                # for i, (data,labels) in enumerate(test_ood_loader):

                #     try:
                #         data1, labels1 = next(dataloader_iterator) # 
                #     except StopIteration:
                #         dataloader_iterator = iter(test_id_loader)
                #         data1, labels1 = next(dataloader_iterator)

                #for i, ((data,labels), (data1, labels1)) in enumerate(zip(test_id_loader, test_ood_loader)):
                #------------------------VAL DATA------------------------------------
                for i, (data, labels) in enumerate(test_id_loader):
                    #print(f' shape of i :{i}, shape of val dataloader: {len(data)}, shape of test_dataloader :{len(data1)}')
                   
                    data, labels = data.cpu(), labels.cpu()
                    dists, out, out_big = model(data)
                    scores = -dists + model.alphas
                    # _, predicted = torch.max(scores, 1)
                    # conf, _ = torch.exp(torch.min(dists, dim=1)) # making exp confidence to bound the conf
                    conf= torch.exp(-torch.min(dists, dim=1).values) # making exp confidence to bound the conf
                    # ----- bounding the confidence they produce and marking ood if exp_conf <= 0.5:
                    predicted= -1 + torch.zeros(len(labels), dtype=torch.int64) # inittializing everything as -1
                    predicted = predicted.cpu()
                    for i in range(len(scores)):
                        if (conf[i] > args.conf): #or (torch.max(scores1[i]) >= 0.5): # --- positive score means within the cluster,
                           predicted[i] = torch.argmax(scores[i])
                    #conf, _ = torch.min(dists, dim=1)
                    for j in range(len(out_big)):
                    #labels_list_test.append([j].squeeze().tolist())
                        #my_out_test_id = out+labels
                        out_feat_test.append(out_big[j][0].squeeze().tolist()+ [labels[j].squeeze().tolist()]+[predicted[j].squeeze().tolist()] + scores[j,:].squeeze().tolist() + [conf[j].squeeze().tolist()])
                    #print(f' shape of val test data :{len(out_feat_test)}, {len(out_feat_test[0])}')
                   
                # print(f"centre from val: {model.centers}")
                # print(f"for validation radii:{radii1}, \n alpha : {param}")

            correct1, total1 = 0, 0
            out_feat_test1 = []
            for i, (data1, labels1) in enumerate(test_ood_loader):
                data1, labels1 = data1.cpu(), labels1.cpu()
                
                #print(f'shape ood of labels :{labels1.shape}, of data {data1.shape}')#, out1.shape)
                dists1, out1, out_big1 = model(data1)
                scores1 = -dists1 + model.alphas
                # _, predicted1 = torch.max(scores1, 1)
                # conf1, _ = torch.min(dists1, dim=1)
                conf1= torch.exp(-torch.min(dists1, dim=1).values) # making exp confidence to bound the conf
                    # ----- bounding the confidence they produce and marking ood if exp_conf <= 0.5:
                predicted1= -1 + torch.zeros(len(labels1), dtype=torch.int64) # inittializing everything as -1
                predicted1 = predicted1.cpu()
                for i in range(len(scores1)):
                    if (conf1[i] > args.conf): #or (torch.max(scores1[i]) >= 0.5): # --- positive score means within the cluster,
                        predicted1[i] = torch.argmax(scores1[i])
                for j in range(len(out_big1)):

                    out_feat_test1.append(out_big1[j][0].squeeze().tolist()+[labels1[j].squeeze().tolist()]+[predicted1[j].squeeze().tolist()]+ scores1[j,:].squeeze().tolist()+ [conf1[j].squeeze().tolist()])
                #print(f' shape of ood test data :{len(out_feat_test1)}, {len(out_feat_test1[0])}')
            # print(f"centre from test: {model.centers}")
            # print(f"for test radii:{radii11}, \n alpha : {param1}")
    

    ###----------saving model -----------------###
    torch.save(model, outdir+'model_saved'+str(args.dataset)+'.pt')

    print(f' shape of complete train data_from dataloader: {len(out_feat_train)}, of validation : {len(out_feat_test)} of ood:{ len(out_feat_test1)}')
    #----trying to store representation  as text files
    outfile = os.path.join(outdir, 'latent_train_representation.csv')
    f = open(outfile, 'w')
    for i in range(len(out_feat_train)):
        f.write("{}\n".format(out_feat_train[i]))

    f.close()

    #----va;lidation ---------
    outfile = os.path.join(outdir, 'latent_val_representation.csv')
    f = open(outfile, 'w')
    for i in range(len(out_feat_test)):
        f.write("{}\n".format(out_feat_test[i]))

    f.close()

    #----test -- ood ---------
    outfile = os.path.join(outdir, 'latent_test_ood_representation.csv')
    f = open(outfile, 'w')
    for i in range(len(out_feat_test1)):
        f.write("{}\n".format(out_feat_test1[i]))

    f.close()

    
    #--------------getting accurtacy ---precison _f1 score and recall------------
    #---------for train data--------------------
    hidden_size = 128
    print(f'accuracy, precison, recall, anf f1 score for TRAIN DATA')
    # Get the confusion matrix
    #directory = 'C:/Users/e117907/OneDrive - Mastercard/Desktop/Semi_supervised/DeepMCDD-master/output/'
    # df = pd.read_csv(directory+ str(args.net_type)+'_'+ str(args.dataset) +'/latent_train_representation.csv')
    df = pd.read_csv(outdir+'/latent_train_representation.csv')
    df.columns =  ['feat_'+str(x+1) for x in range(len(df.columns))]
    
    col = list(df.columns)
    print(f'value count for train label {df[col[hidden_size]].value_counts()}')
    print(f'value count for train predicted {df[col[hidden_size + 1]].value_counts()}')
    label = list(df[col[hidden_size]].unique())
    label.sort()
    cm = confusion_matrix(np.array(df.iloc[:, hidden_size]), np.array(df.iloc[:, hidden_size+1]), labels = label)
    print(cm
    )
    # We will store the results in a dictionary for easy access later
    per_class_accuracies = {}
    for idx in range(len(label)):
        true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
        true_positives = cm[idx, idx]
        per_class_accuracies[idx] = (true_positives + true_negatives) / np.sum(cm)

    a = precision_recall_fscore_support(np.array(df.iloc[:, hidden_size]), np.array(df.iloc[:, hidden_size+1]), average=None)
    # label = ['ethoca_fpf', 'tpf', 'auth_undisp']
    for i in range(len(a[0])):
        print(f'for class {label[i]}: \ntotal number of actual samples: {a[3][i]}, \n accuracy : {per_class_accuracies[i]} ,\nprecision: {a[0][i]},\n recall : {a[1][i]}\n and f1 score {a[2][i]} \n')

       #  #=========for test data===============
    print(f'value count for indistribution and OOD TEST DATA ')
    print(f'accuracy, precison, recall, anf f1 score for TEST DATA ')
    # Get the confusion matrix
    #print(args.dataset)
    #locatio_n = '/output/'+str(args.dataset)
    df1 = pd.read_csv(outdir + '/latent_val_representation.csv')
    df1.columns =  ['feat_'+str(x+1) for x in range(len(df1.columns))]
    df2 = pd.read_csv(outdir + '/latent_test_ood_representation.csv')
    df2.columns =  ['feat_'+str(x+1) for x in range(len(df2.columns))]
    print(f"val shape {df1.shape}, ood shape {df2.shape} ")
    df = pd.concat([df1, df2])
    col = list(df1.columns)
    print(f'value count for test label {df[col[hidden_size]].value_counts()}')
    print(f'value count for test predicted {df[col[hidden_size + 1]].value_counts()}')
    label = list(df[col[hidden_size]].unique())
    label.sort()
    cm = confusion_matrix(np.array(df.iloc[:, hidden_size]), np.array(df.iloc[:, hidden_size+1]),labels = label)
    print(cm)
    # We will store the results in a dictionary for easy access later
    per_class_accuracies = {}
    for idx in range(len(label)):
        true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
        true_positives = cm[idx, idx]
        per_class_accuracies[idx] = (true_positives + true_negatives) / np.sum(cm)

    a = precision_recall_fscore_support(np.array(df.iloc[:,hidden_size]), np.array(df.iloc[:, hidden_size+1]), average=None)
    # label = ['ethoca_fpf', 'tpf', 'auth_undisp']
    for i in range(len(a[0])):
        print(f'for class {label[i]}: \ntotal number of actual samples: {a[3][i]}, \n accuracy : {per_class_accuracies[i]} ,\nprecision: {a[0][i]},\n recall : {a[1][i]}\n and f1 score {a[2][i]} \n')
    print("overall classification metric validation ")
    p,r,f,supp = precision_recall_fscore_support(np.array(df1.iloc[:,hidden_size]), np.array(df1.iloc[:, hidden_size+1]), average='weighted')
    acc = accuracy_score(np.array(df1.iloc[:,hidden_size]), np.array(df1.iloc[:, hidden_size+1]), normalize = True)
    print(f" overall accuracy : {acc} , \n weighted precision: {p},\n weighted recall: {r},\n weighted f1-score: {f}\n")
    print("overall classification metric in-distribution along with ood")
    p,r,f,supp = precision_recall_fscore_support(np.array(df.iloc[:,hidden_size]), np.array(df.iloc[:, hidden_size+1]), average='weighted')
    acc = accuracy_score(np.array(df.iloc[:,hidden_size]), np.array(df.iloc[:, hidden_size+1]), normalize = True)
    print(f" overall accuracy : {acc} , \n weighted precision: {p},\n weighted recall: {r},\n weighted f1-score: {f}\n")
    print(f'for class ood: \ntotal number of actual samples: {a[3][0]}, \n accuracy : {per_class_accuracies[0]} ,\nprecision: {a[0][0]},\n recall : {a[1][0]}\n and f1 score {a[2][0]} \n')
    
if __name__ == '__main__':
    main()
