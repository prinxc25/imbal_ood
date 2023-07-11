import os, argparse
import torch
import numpy as np
import itertools
import models
import sklearn
import pandas as pd
import torch.nn.functional as f
from sklearn.metrics import precision_recall_fscore_support
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
parser.add_argument('--conf',type=float,default = 0.7, help='conf for making prediction in dataset')
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
parser.add_argument('--reg_lambda', type=float, default=1.0, help='regularization coefficient')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')

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
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(args.gpu)

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
        model.cuda()
       #class MLP_DeepMCDD(nn.Module):
       #def __init__(self, input_size, hidden_sizes, num_classes)
       # latent_size = dimension size for latent representation, default = 128
       # num_layers = the number of hidden layers in MLP, default = 3
        ce_loss = torch.nn.CrossEntropyLoss()
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
                
                data, labels = data.cuda(), labels.cuda()

                dists, out, out_big = model(data) 
               
                radii1 = torch.exp(model.logsigmas)
                param = torch.exp(model.param)
                #scores = - dists + model.alphas #----
                scores = - torch.abs(dists) + torch.mul(param,radii1) #----if it was in cluster score with be positive otherwise negative  ---so in prediction we take max
                #conf, _ = torch.max(scores, dim=1)
                #print(scores)
                #_, predicted = torch.argmax(scores, 1)
                predicted = torch.argmax(scores, 1)
                #print(f'shape train of out feature {out_big.shape}, of predicted {predicted.shape}')
                ##---way 1-----------
                # scores_12 = torch.exp(torch.div(scores, torch.mul(param,radii1))-1)
                ## way 2----------------------
                scores_12 = torch.exp(0.65*(torch.div(scores, torch.mul(param,radii1))-1)) + torch.exp(10**(-12))
                # multiplying buy 0.69 so that at circumference prob is a little more than 0.5
                # adding torch.exp(10**(-12)) for stability
                scores_12 = f.normalize(scores_12 , p=2, dim=1)#--- row normalizing for scores to be of probability nature
                conf, _ = torch.max(scores_12, dim=1)
                label_mask = torch.zeros(labels.size(0), model.num_classes).cuda().scatter_(1, labels.unsqueeze(dim=1), 1)
                #print(label_mask)
                #my_out = torch.mul(label_mask, (dists+centr))
                my_out = torch.mul(label_mask,dists)
                pull_loss1 = torch.mean(torch.sum(torch.mul(label_mask, dists), dim=1))
                push_loss1 = torch.mean(torch.sum(torch.mul(1-label_mask, torch.div(args.reg_lambda,  dists)), dim=1))
                #push_loss1 = torch.mean(torch.sum(torch.mul(1-label_mask, torch.div(1, torch.mul(torch.abs(model.alphas), dists))), dim=1)) #---this is working fine as of now -- push lloss
                pull_loss2 = ce_loss(scores_12, labels)#---the probem is happening here----scores are negative tooo -- pull loss
                #loss = args.reg_lambda * pull_loss + push_loss 
                loss =  pull_loss1 + push_loss1 + pull_loss2 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # total_data = total_data+ len(labels)
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
                   
                    data, labels = data.cuda(), labels.cuda()
                    
                    dists, out, out_big = model(data)
                    #conf, _ = torch.min(dists, dim=1)
                    #print(f'confidence shape:{conf.shape}')
                    centers_c = model.centers
                    #scores = - dists + model.alphas
                    radii1 = radii1
                    param = param
                    scores = - torch.abs(dists) + torch.mul(param,radii1) #----if it was in cluster score with be positive otherwise negative  ---so in prediction we take max
                    # scores_12 = torch.exp(torch.div(scores, torch.mul(param,radii1))-1)
                    ## way 2----------------------
                    scores_12 = torch.exp(0.65*(torch.div(scores, torch.mul(param,radii1))-1)) + torch.exp(10**(-12))
                    # multiplying buy 0.69 so that at circumference prob is a little more than 0.5
                    # adding torch.exp(10**(-12)) for stability
                    scores_12 = f.normalize(scores_12 , p=2, dim=1)#--- row normalizing for scores to be of probability nature
                    conf, _ = torch.max(scores_12, dim=1)
                    #conf, _ = torch.max(scores, dim=1)
                    # _, predicted = torch.argmax(scores, 1)
                    predicted = torch.argmax(scores, 1)
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
                data1, labels1 = data1.cuda(), labels1.cuda()
                #print(f'shape ood of labels :{labels1.shape}, of data {data1.shape}')#, out1.shape)
                dists1, out1, out_big1 = model(data1)
                #conf1, _ = torch.min(dists1, dim=1)
                #print(f'confidence shape:{conf1.shape}')
                #centers_c1 = model.centers
                #scores1 = - dists1 + model.alphas
                radii11 = radii1
                param1 = param
                scores1 = -torch.abs(dists1) +torch.mul(param1,radii11) #----if it was in cluster score with be positive otherwise negative  ---so in prediction we take max
                # scores_121 = torch.exp(torch.div(scores1, torch.mul(param1,radii11))-1)
                ## way 2----------------------
                scores_121 = torch.exp(0.65*(torch.div(scores, torch.mul(param1,radii11))-1)) + torch.exp(10**(-12))
                # multiplying buy 0.69 so that at circumference prob is a little more than 0.5
                # adding torch.exp(10**(-12)) for stability
                scores_121 = f.normalize(scores_121 , p=2, dim=1)#--- row normalizing for scores to be of probability nature
                conf1, _ = torch.max(scores_121, dim=1)
                #conf1, _ = torch.max(scores1, dim=1)
                #_, predicted1 = torch.max(scores1, 1)
                predicted1= -1 + torch.zeros(len(labels1), dtype=torch.int64) # inittializing everything as -1
                predicted1 = predicted1.cuda()
                for i in range(len(scores1)):
                    if ((torch.max(scores1[i]) >= 0) & (conf1[i] > args.conf)): #or (torch.max(scores1[i]) >= 0.5): # --- positive score means within the cluster,
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

    # -----------for val data -----------ique
    print(f'accuracy, precison, recall, anf f1 score for VALIDATION DATA ')
    # Get the confusion matrix
    #print(args.dataset)
    #locatio_n = '/output/'+str(args.dataset)
    df = pd.read_csv(outdir + '/latent_val_representation.csv')
    df.columns =  ['feat_'+str(x+1) for x in range(len(df.columns))]
    col = list(df.columns)
    print(f'value count for val label {df[col[hidden_size]].value_counts()}')
    print(f'value count for val predicted {df[col[hidden_size + 1]].value_counts()}')
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


    #=========for test data===============
    print(f'value count for  TEST DATA ')
    # Get the confusion matrix
    #print(args.dataset)
    #locatio_n = '/output/'+str(args.dataset)
    #df = pd.read_csv(directory+str(args.net_type)+'_'+  str(args.dataset) +'/latent_test_ood_representation.csv')
    df= pd.read_csv(outdir+'/latent_test_ood_representation.csv')
    df.columns =  ['feat_'+str(x+1) for x in range(len(df.columns))]
    col = list(df.columns)
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
    print(f'value count for ood test data points: \n {df[df[col[hidden_size]]== -1][col[hidden_size + 1]].value_counts()}')
    
    # #         #---we want to save 
    #         best_idacc = max(idacc_list)
    #         best_oodacc = oodacc_list[idacc_list.index(best_idacc)]
            
    #         print('== {fidx:1d}-th fold results =='.format(fidx=fold_idx+1))
    #         print('The best ID accuracy on "{idset:s}" test samples : {val:6.2f}'.format(idset=args.dataset, val=best_idacc))
    #         print('The best OOD accuracy on "{oodset:s}" test samples :'.format(oodset=args.dataset+'_'+str(args.oodclass_idx)))
    #         print_ood_results(best_oodacc)
    #         #print()
    #         best_idacc_list.append(best_idacc)
    #         best_oodacc_list.append(best_oodacc)
    #     #print(f'{out[0,0]},\n {out[0,0].shape}')
    #     #x_n = np.array(out[0])
    #     #y_n = np.array(out[1])
    #     #a = np.allclose(x, y)
    #     # for i in range(63):
    #     #     print(torch.equal(data[0], data[i+1]))
    #     #     print(torch.equal(out[0], out[i+1]))
    #     #print(out[0]==out[1])
    #     print('== Final results ==')
    #     print('The best ID accuracy on "{idset:s}" test samples : {mean:6.2f} ({std:6.3f})'.format(idset=args.dataset, mean=np.mean(best_idacc_list), std=np.std(best_idacc_list)))
    #     print('The best OOD accuracy on "{oodset:s}" test samples :'.format(oodset='class_'+str(args.oodclass_idx)))
    #     print_ood_results_total(best_oodacc_list)
 

if __name__ == '__main__':
    main()
