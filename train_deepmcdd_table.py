import os, argparse
import torch
import numpy as np
import torch.nn.functional as F
import models
from dataloader_table import get_table_data, get_table_data_with_iden_separate_sets, get_table_data_train_test, get_table_data_train_test_minority,\
    get_table_data_train_test_minority_creation
from utils import  compute_metrics, print_ood_results, print_ood_results_total,compute_metrics_ours_minority, \
    get_mahalanobis_distance, compute_confscores_mhnb_minority#, print_minority_results_total


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='gas | shuttle | drive | mnist')
parser.add_argument('--net_type', required=True, help='mlp')
# parser.add_argument('--num_classes', required=True,type=int, help='number of classes including in dist and ood')
# parser.add_argument('--num_features', required=True,type=int, help='number of features in dataset')
parser.add_argument('--extended_name', default='', help='extended name you you wan to add to the output')
parser.add_argument('--conf', type=float, default=0.9, help='confidence for marking id')
parser.add_argument('--model_path', default=0, help='path to saved model')
parser.add_argument('--datadir', default='./table_data/', help='path to dataset')
parser.add_argument('--outdir', default='./output/', help='folder to output results')
parser.add_argument('--oodclass_idx', type=int,required=True, help='index of the OOD class')
parser.add_argument('--minority_class_idx', type=int,required=True, help='index of the minority class')
parser.add_argument('--downsample_ratio', type=float, default=0.30, help='downsample_ratio for making minority class')
# parser.add_argument('--classes_id', type=int,required=True, help='total training classes ')
parser.add_argument('--batch_size', type=int, default=1000, help='batch size for data loader')
parser.add_argument('--latent_size', type=int, default=128, help='dimension size for latent representation') 
parser.add_argument('--num_layers', type=int, default=3, help='the number of hidden layers in MLP')
parser.add_argument('--num_folds', type=int, default=1, help='the number of cross-validation folds')
parser.add_argument('--num_epochs', type=int, default=10, help='the number of epochs for training sc-layers')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate of Adam optimizer')
parser.add_argument('--reg_pull', type=float, default=1.0, help='regularization coefficient for min dist')
parser.add_argument('--gpu', type=int, default=-1, help='gpu index')
args = parser.parse_args()
print(args)

def main():
    outdir = os.path.join(args.outdir, args.net_type + '_' + args.extended_name+'_mhnb_' + args.dataset)

    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # torch.cuda.set_device(args.gpu)

    best_idacc_list, best_oodacc_list = [], []
    # best_classesacc_list = [[] for _ in range(args.classes_id)]
    best_classesacc_list =[]
    for fold_idx in range(args.num_folds):
        
        # train_loader, test_id_loader, test_ood_loader, num_features1, num_classes1 = get_table_data_with_iden_separate_sets(args.batch_size, args.datadir, args.dataset, args.oodclass_idx, fold_idx)
        train_loader, test_id_loader, test_ood_loader, num_features1, num_classes1, class_mapping = get_table_data_train_test_minority_creation(args,fold_idx)
        
        # train_loader, test_id_loader, test_ood_loader, num_features1, num_classes1, class_mapping = get_table_data_train_test_minority(args.batch_size, args.datadir, args.dataset, args.oodclass_idx, fold_idx)
        
        print(f' shape of complete train data: {len(train_loader.dataset)}, of validation : {len(test_id_loader.dataset)} of ood:{ len(test_ood_loader.dataset)}')
        print(f'class mapped to key where  original class is vlaue[0] and value count is vlaue[1] for classes in test data :\n {class_mapping}')
        # if args.dataset == 'gas':
        #     num_classes, num_features = 6, 128
        # elif args.dataset == 'drive':
        #     num_classes, num_features = 11, 48
        # elif args.dataset == 'shuttle':
        #     num_classes, num_features = 7, 9
        # elif args.dataset == 'mnist':
        #     num_classes, num_features = 10, 784
        # else: 
        desired_value = args.minority_class_idx
        desired_key = None
        for key, value in class_mapping.items():
            if value[0] == desired_value:
                desired_key = key
                break 
        num_classes, num_features = num_classes1, num_features1 
        print(f' number of classes : {num_classes}, number of features: {num_features}, no of id classes: {num_classes-1}' )
        print(f"key to minority class is : { desired_key } and it's corresponding value is : {class_mapping[ desired_key ]}")
        model = models.MLP_MNB(num_features, args.num_layers*[args.latent_size], num_classes=num_classes-1)
        model.cpu()

        ce_loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
   
        idacc_list, oodacc_list = [], []
        # classesacc_list = [[] for _ in range(args.classes_id)]
        classesacc_list = []
        total_step = len(train_loader)
        for epoch in range(args.num_epochs):
            model.train()
            total_loss = 0.0
            
            train_embeds=[]
            train_labels=[]
             
            for i, (data, labels) in enumerate(train_loader):
                data, labels = data.cpu(), labels.cpu()
                
                scores, embed = model(data) 
                scores=-1*get_mahalanobis_distance(embed,labels,embed)

                label_mask = torch.zeros(labels.size(0), model.num_classes).cpu().scatter_(1, labels.unsqueeze(dim=1), 1)

                push_loss = ce_loss(scores, labels)
                loss = push_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                total_loss += loss.item()
            
                train_embeds.append(embed)
                train_labels.append(labels)
                
            train_embeds=torch.cat(train_embeds,dim=0)
            train_labels=torch.cat(train_labels,dim=0)
            print(f'loss for epoch {epoch} is : {loss}')

            model.eval()
            with torch.no_grad():
                # (1) evaluate ID classification
                correct, total = 0, 0
                for data, labels in test_id_loader:
                    scores, embed = model(data)
                    scores=get_mahalanobis_distance(embed,train_labels,train_embeds)
                    _, predicted = torch.min(scores, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                idacc_list.append(100 * correct / total)
                
                # (2) evaluate OOD detection
                # compute_confscores_ours_minority(model, test_loader, outdir, id_flag, minority_flag)
                compute_confscores_mhnb_minority(model, test_id_loader, train_labels, train_embeds, outdir, True, -2, args)
                compute_confscores_mhnb_minority(model, test_ood_loader, train_labels, train_embeds, outdir, False, -2, args)
                oodacc_list.append(compute_metrics_ours_minority(outdir, f'confscores_{args.dataset}_id.txt', f'confscores_{args.dataset}_ood.txt',-2))
                # for class_idx in range(args.classes_id):
                #     compute_confscores_theirs_minority(model, test_id_loader, outdir, False, class_idx, args)
                #     classesacc_list[class_idx].append(compute_metrics_ours_minority(outdir,\
                #           f'confscores_{args.dataset}_class_{class_idx}.txt', f'confscores_{args.dataset}_rest_class_{class_idx}.txt', class_idx))
                compute_confscores_mhnb_minority(model, test_id_loader,train_labels, train_embeds,  outdir, False, desired_key, args)
                classesacc_list.append(compute_metrics_ours_minority(outdir,\
                        f'confscores_{args.dataset}_class_{ desired_key }.txt', f'confscores_{args.dataset}_rest_class_{ desired_key }.txt',  desired_key ))

                    # compute_metrics_ours_minority(dir_name, known_file, novel_file, verbose=False))

        best_idacc = max(idacc_list)
        best_oodacc = oodacc_list[idacc_list.index(best_idacc)]
        best_classesacc =  classesacc_list[idacc_list.index(best_idacc)]
        # best_classesacc =  [[] for _ in range(args.classes_id)]

        # print(f"best_idacc : {best_idacc},\n idacc_list : {idacc_list}, \n  oodacc_list : {oodacc_list}\n \
        #      best_oodacc : {best_oodacc}, \n classesacc_list : {classesacc_list}")
        # best_minorityacc = minorityacc_list[idacc_list.index(best_idacc)]
        # for class_idx in range(args.classes_id):
        #     best_classesacc[class_idx] = classesacc_list[class_idx][idacc_list.index(best_idacc)]
        
        print('== {fidx:1d}-th fold results =='.format(fidx=fold_idx+1))
        print('The best ID accuracy on "{idset:s}" test samples : {val:6.2f}'.format(idset=args.dataset, val=best_idacc))
        # print('The best OOD accuracy on "{oodset:s}" test samples {countood:d} :'.format(oodset=args.dataset+'_'+str(args.oodclass_idx), countood=some_variable))

        print('The best OOD accuracy on "{oodset:s}" test samples {countood:d} :'.format(oodset=args.dataset+'_'+str(args.oodclass_idx),\
             countood = class_mapping[-1][1]))
        print_ood_results(best_oodacc)
        print('The best accuracy in-distribution on class "{classesset:s}" test samples {countclass:d}:'.format(classesset=args.dataset+'_'+\
                str(class_mapping[ desired_key ][0]), countclass = class_mapping[ desired_key ][1]))
        print_ood_results(best_classesacc)        
        # for class_idx in range(args.classes_id):
        #     # class_idx is the mapped new class and inside class_mappinf it cointisn the name of original class with its vaue count
        #     print('The best accuracy in-distribution on class "{classesset:s}" test samples {countclass:d}:'.format(classesset=args.dataset+'_'+\
        #         str(class_mapping[class_idx][0]), countclass = class_mapping[class_idx][1]))
        #     print_ood_results(best_classesacc[class_idx])

        best_idacc_list.append(best_idacc)
        best_oodacc_list.append(best_oodacc)
        # for class_idx in range(args.classes_id):
        #     best_classesacc_list[class_idx].append(best_classesacc[class_idx])
        best_classesacc_list.append(best_classesacc)
    # bestres =   pd.DataFrame(columns = [])     
    print('== Final results ==')
    print('The best ID accuracy on "{idset:s}" test samples : {mean:6.2f} ({std:6.3f})'.format(idset=args.dataset, mean=np.mean(best_idacc_list), std=np.std(best_idacc_list)))
    print('The best OOD accuracy on "{oodset:s}" test samples :'.format(oodset='class_'+str(args.oodclass_idx)))
    print_ood_results_total(best_oodacc_list)
    # for class_idx in range(args.classes_id):
    #     print('The best accuracy on in-distribution class "{classesset:s}" test samples {countclass:d}:'.format(classesset=args.dataset+'_'+\
    #         str(class_mapping[class_idx][0]), countclass = class_mapping[class_idx][1]))
    #     # print('The best  accuracy on  class "{classesset:s}" test samples :'.format(classesset='class_'+str(class_idx)))
    #     print_minority_results_total(best_classesacc_list[class_idx])
    print('The best accuracy on in-distribution class "{classesset:s}" test samples {countclass:d}:'.format(classesset=args.dataset+'_'+\
            str(class_mapping[ desired_key ][0]), countclass = class_mapping[ desired_key ][1]))
    print_ood_results_total(best_classesacc_list)        

if __name__ == '__main__':
    main()
#  time python TD_ours_minority.py --dataset gas --net_type mlp --num_layers 3  --num_epochs 100  --num_folds 5 --oodclass_idx 0 --classes_id 5 --minority_class_idx 2 --reg_push 1 --reg_another 1 --reg_pull 1 --reg_focal1 1 --reg_focal2 0 --downsample_ratio 1 --extended_name not_focal2   
