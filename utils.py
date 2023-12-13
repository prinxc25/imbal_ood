#
# This file is implemented based on the author code of
#    Lee et al., "A simple unified framework for detecting out-of-distribution samples and adversarial attacks", in NeurIPS 2018.
#

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, balanced_accuracy_score


def compute_confscores(model, test_loader, outdir, id_flag):
    total = 0
    if id_flag == True:
        outfile = os.path.join(outdir, 'confscores_id.txt')
    else:
        outfile = os.path.join(outdir, 'confscores_ood.txt')

    f = open(outfile, 'w')
    
    for data, _ in test_loader:
        # dists = model(data.cuda())
        dists, out, out_big = model(data.cuda())
        confscores, _ = torch.min(dists, dim=1)
        total += data.size(0)

        for i in range(data.size(0)):
            f.write("{}\n".format(-confscores[i]))
    
    f.close()

def get_auroc_curve(indir):
    # known = np.loadtxt(os.path.join(indir, 'confscores_id.txt'), delimiter='\n')

    # novel = np.loadtxt(os.path.join(indir, 'confscores_ood.txt'), delimiter='\n')
    known = np.loadtxt(os.path.join(indir, 'confscores_id.txt'))#, delimiter='\n')
    
    novel = np.loadtxt(os.path.join(indir, 'confscores_ood.txt'))#, delimiter='\n')
    known.sort()
    novel.sort()
    
    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])
    
    num_k = known.shape[0]
    num_n = novel.shape[0]
    
    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]
    tpr85_pos = np.abs(tp / num_k - .85).argmin()
    tpr95_pos = np.abs(tp / num_k - .95).argmin()
    tnr_at_tpr85 = 1. - fp[tpr85_pos] / num_n
    tnr_at_tpr95 = 1. - fp[tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr85, tnr_at_tpr95

def compute_metrics(dir_name, verbose=False):
    tp, fp, tnr_at_tpr85, tnr_at_tpr95 = get_auroc_curve(dir_name)
    results = dict()
    mtypes = ['TNR85', 'TNR95', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')
        
    if verbose:
        print('{stype:5s} '.format(stype=stype), end='')
    results = dict()
    
    # TNR85
    mtype = 'TNR85'
    results[mtype] = tnr_at_tpr85
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')

    # TNR95
    mtype = 'TNR95'
    results[mtype] = tnr_at_tpr95
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
    
    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1. - fpr, tpr)
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
    
    # DTACC
    mtype = 'DTACC'
    results[mtype] = .5 * (tp/tp[0] + 1. - fp/fp[0]).max()
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
    
    # AUIN
    mtype = 'AUIN'
    denom = tp + fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
    
    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0] - tp + fp[0] - fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0] - fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
        print('')

    return results

def print_ood_results(ood_result):

    for mtype in ['TNR85', 'TNR95', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('\n{val:6.2f}'.format(val=100.*ood_result['TNR85']), end='')
    print(' {val:6.2f}'.format(val=100.*ood_result['TNR95']), end='')
    print(' {val:6.2f}'.format(val=100.*ood_result['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100.*ood_result['DTACC']), end='')
    print(' {val:6.2f}'.format(val=100.*ood_result['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100.*ood_result['AUOUT']), end='')
    print('')

def print_ood_results_total(ood_result_list):
    
    TNR85_list = [100.*ood_result['TNR85'] for ood_result in ood_result_list]
    TNR95_list = [100.*ood_result['TNR95'] for ood_result in ood_result_list]
    AUROC_list = [100.*ood_result['AUROC'] for ood_result in ood_result_list]
    DTACC_list = [100.*ood_result['DTACC'] for ood_result in ood_result_list]
    AUIN_list = [100.*ood_result['AUIN'] for ood_result in ood_result_list]
    AUOUT_list = [100.*ood_result['AUOUT'] for ood_result in ood_result_list]

    for mtype in ['TNR85', 'TNR95', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']:
        print(' {mtype:15s}'.format(mtype=mtype), end='')
    print('\n{mean:6.2f} ({std:6.3f})'.format(mean=np.mean(TNR85_list), std=np.std(TNR85_list)), end='')
    print(' {mean:6.2f} ({std:6.3f})'.format(mean=np.mean(TNR95_list), std=np.std(TNR95_list)), end='')
    print(' {mean:6.2f} ({std:6.3f})'.format(mean=np.mean(AUROC_list), std=np.std(AUROC_list)), end='')
    print(' {mean:6.2f} ({std:6.3f})'.format(mean=np.mean(DTACC_list), std=np.std(DTACC_list)), end='')
    print(' {mean:6.2f} ({std:6.3f})'.format(mean=np.mean(AUIN_list), std=np.std(AUIN_list)), end='')
    print(' {mean:6.2f} ({std:6.3f})\n'.format(mean=np.mean(AUOUT_list), std=np.std(AUOUT_list)), end='')
    print('')


def save_and_evaluate_at_epoch(epoch, model, out_feat_train, out_feat_test, out_feat_test1, outdir):
    # Saving the model
    model_filename = f'model_saved_epoch_{epoch}.pt'
    torch.save(model, os.path.join(outdir, model_filename))

    # Saving representations as text files
    save_representation(out_feat_train, 'latent_train_representation.csv', outdir)
    save_representation(out_feat_test, 'latent_val_representation.csv', outdir)
    save_representation(out_feat_test1, 'latent_test_ood_representation.csv', outdir)

    # Evaluating classification metrics
    df1 = pd.DataFrame(out_feat_train, columns=[f'feat_{x+1}' for x in range(len(out_feat_train[0]))])
    evaluate_classification_metrics(df1, 'Train', outdir)

    # evaluate_classification_metrics(out_feat_test, 'Validation', outdir)
    df2 = pd.DataFrame(out_feat_test, columns=[f'feat_{x+1}' for x in range(len(out_feat_test[0]))])
    df3 = pd.DataFrame(out_feat_test1, columns=[f'feat_{x+1}' for x in range(len(out_feat_test1[0]))])

    evaluate_classification_metrics(pd.concat([df2, df3]), 'Validation and OOD', outdir)
    evaluate_classification_metrics(df3, 'Test OOD', outdir)

def save_representation(features, filename, outdir):
    outfile = os.path.join(outdir, filename)
    with open(outfile, 'w') as f:
        for i in range(len(features)):
            f.write("{}\n".format(features[i]))

def evaluate_classification_metrics(features, dataset_type, outdir):
    hidden_size = 128
    print(f'Accuracy, precision, recall, and F1 score for {dataset_type} DATA')
    
    # df = pd.DataFrame(features, columns=[f'feat_{x+1}' for x in range(len(features[0]))])
    df = features
    col = list(df.columns)
    print(f"{dataset_type} {df.shape}")
    print(f'value count for test label {df[col[hidden_size]].value_counts()}')
    print(f'value count for test predicted {df[col[hidden_size + 1]].value_counts()}')
    label = list(df[col[hidden_size]].unique())
    label.sort()
    if dataset_type != 'Test OOD':
        # Code for confusion matrix and per-class accuracies...
        cm = confusion_matrix(np.array(df.iloc[:, hidden_size]), np.array(df.iloc[:, hidden_size+1]),labels = label)
        print(cm)
        # Code for precision, recall, and F1 score...
        # We will store the results in a dictionary for easy access later
    
        per_class_accuracies = {}
        for idx in range(len(label)):
            true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
            true_positives = cm[idx, idx]
            per_class_accuracies[idx] = (true_positives + true_negatives) / np.sum(cm)

        a = precision_recall_fscore_support(np.array(df.iloc[:,hidden_size]), np.array(df.iloc[:, hidden_size+1]), average=None)
        # label = ['ethoca_fpf', 'tpf', 'auth_undisp']
        for i in range(len(a[0])):
            print(f'for {dataset_type}  class {label[i]}: \ntotal number of actual samples: {a[3][i]}, \n accuracy : {per_class_accuracies[i]} ,\nprecision: {a[0][i]},\n recall : {a[1][i]}\n and f1 score {a[2][i]} \n')
          
    # if dataset_type != 'Train':   
    #     # # Check if the file exists
    #     if os.path.exists(file_path):
    #         print(f"The file '{file_name}' exists in the directory.")
      
    #     # col = ['file_name','data','num_layers', 'num_epoch', 'metrics_type', 'accuracy', 'precision', 'recall',
    #     #    'f1_score', 'supp',  'reg_push', 'reg_pull',
    #     #    'reg_focal1', 'focal_gamma1', 'reg_focal2', 'focal_gamma2',
    #     #    'reg_another']
    #     # dfacc = pd.DataFrame(columns=col)
    #     # print(supp)
    #     dfacc = pd.read_csv('accuracy_for_various_models.csv', index_col=0)# ---- for adding everything in a csv
    #     """
    #     In macro averaging, you calculate the metric independently for each class and then take the average.
    #     Each class is given equal weight in the computation of the average.
    #     This means that the metric is not biased towards the majority class and treats all classes equally.
    #     In weighted averaging, you calculate the metric for each class, but the contribution of each class
    #     is weighted by its support (the number of true instances).
    #     This means that classes with more instances have a greater impact on the average.
    #     """
    #     another = 'another_scores_0'
    #     new_row = {'file_name':another, 'metrics_type':'validation', 'data':args.dataset, 'accuracy':acc, 'precision':p,
    #       'recall':r,'f1_score':f, 'supp':df1.shape[0], 'num_layers': args.num_layers, 'num_epoch': args.num_epochs, 'reg_push':args.reg_push,
    #         'reg_pull':args.reg_pull,'reg_focal1':args.reg_focal1, 'focal_gamma1':args.focal_gamma1, 'reg_focal2':args.reg_focal2,
    #         'focal_gamma2':args.focal_gamma2,
    #       'reg_another':args.reg_another}
    #     # Append the dictionary as a row to the DataFrame
    #     dfacc = dfacc.append(new_row, ignore_index=True)
  

    print(f"overall classification {dataset_type} ")
    # Code for overall classification metrics...
    p,r,f,supp = precision_recall_fscore_support(np.array(df.iloc[:,hidden_size]), np.array(df.iloc[:, hidden_size+1]), average='micro')
    acc = balanced_accuracy_score(np.array(df.iloc[:,hidden_size]), np.array(df.iloc[:, hidden_size+1]))#, normalize = True)
    print(f" overall accuracy : {acc} , \n weighted precision: {p},\n weighted recall: {r},\n weighted f1-score: {f}\n")
    
