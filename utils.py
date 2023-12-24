#
# This file is implemented based on the author code of
#    Lee et al., "A simple unified framework for detecting out-of-distribution samples and adversarial attacks", in NeurIPS 2018.
#

import os
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader

def compute_confscores(model, test_loader, outdir, id_flag):
    total = 0
    if id_flag == True:
        outfile = os.path.join(outdir, 'confscores_id.txt')
    else:
        outfile = os.path.join(outdir, 'confscores_ood.txt')

    f = open(outfile, 'w')
    
    for data, _ in test_loader:
        dists = model(data.cpu())
        confscores, _ = torch.min(dists, dim=1)
        total += data.size(0)

        for i in range(data.size(0)):
            f.write("{}\n".format(-confscores[i]))
    
    f.close()
# def compute_confscores_sftmx(model, test_loader, outdir, id_flag):
#     total = 0
#     if id_flag == True:
#         outfile = os.path.join(outdir, 'confscores_id.txt')
#     else:
#         outfile = os.path.join(outdir, 'confscores_ood.txt')

#     f = open(outfile, 'w')
    
#     for data, _ in test_loader:
#         scores = model(data.cpu())
#         scores = torch.nn.functional.softmax(scores)
#         confscores, _ = torch.max(scores, 1)
#         total += data.size(0)

#         for i in range(data.size(0)):
#             f.write("{}\n".format(confscores[i]))
    
#     f.close()
    
# def compute_confscores_mhlnbs(model,ttl_labels,ttl_embeds, test_loader, outdir, id_flag):
#     total = 0
#     if id_flag == True:
#         outfile = os.path.join(outdir, 'confscores_id.txt')
#     else:
#         outfile = os.path.join(outdir, 'confscores_ood.txt')

#     f = open(outfile, 'w')
    
#     for data, _ in test_loader:
        
#         embed = model(data.cpu())
#         scores=get_mahalnobis_distance(embed,ttl_labels,ttl_embeds)
#         confscores, _ = torch.min(scores, 1)
#         total += data.size(0)

#         for i in range(data.size(0)):
#             f.write("{}\n".format(-confscores[i]))
    
#     f.close()
# import torch

def calculate_class_covariance(embed_k):
    N_k = embed_k.shape[0]
    mean_k = torch.mean(embed_k, dim=0, keepdim=True)
    # mean_per_class.append(mean_k)
    centered_k = embed_k - mean_k
    cov_k = torch.matmul(centered_k.t(), centered_k)
    return cov_k

def calculate_pooled_covariance(embeds_per_class):
    C = len(embeds_per_class)
    N_total = sum(embed.shape[0] for embed in embeds_per_class)

    cov_sum = sum(calculate_class_covariance(embed) for embed in embeds_per_class)
    pooled_covariance = cov_sum / N_total

    return pooled_covariance

def get_mahalanobis_distance(embed, ttl_labels, ttl_embed):
    classes = torch.unique(ttl_labels)
    dists = []
    embeds_per_class = []
    # mean_per_class = []

    # Calculate the pooled covariance matrix
    pooled_cov = calculate_pooled_covariance([ttl_embed[ttl_labels == k] for k in classes])

    # Calculate the inverse of the pooled covariance matrix
    inv_pooled_cov = torch.linalg.inv(pooled_cov)

    # # Get the mean vectors per class from the previous calculation
    mean_per_class = calculate_class_covariance(ttl_embed)

    for k in classes:
        # embed_k = ttl_embed[ttl_labels == k]

        # Mahalanobis distance calculation for the current class
        mahalanobis_term_k = torch.matmul((embed - mean_per_class[k]), inv_pooled_cov)
        mahalanobis_term_k = torch.sum(mahalanobis_term_k * (embed - mean_per_class[k]), dim=1, keepdim=True)

        dists.append(mahalanobis_term_k)

    return torch.cat(dists, dim=1)


#-------------------- next 1 -----------------------------------
def get_mahalnobis_distance_1(embed, ttl_labels, ttl_embed):
    classes = torch.unique(ttl_labels)
    dists = []

    for k in classes:
        embed_k = ttl_embed[ttl_labels == k]
        mean_k = torch.mean(embed_k, 0)
        """variance Calculation: The covariance matrix cov_k should be calculated using unbiased estimates of covariance.
         The torch.cov function may use biased estimates by default. You can set bias=False to get unbiased estimates
        """
        # cov_k = torch.cov(embed_k.T, bias=False)
        cov_k = torch.matmul((embed_k - mean_k).T, (embed_k - mean_k)) / (len(embed_k) - 1)
        inv_cov_k = torch.linalg.inv(cov_k)
        """
        Matrix Multiplication: The expression torch.matmul(embed, inv_cov_k) * embed involves element-wise multiplication,
         which may not be the correct way to calculate Mahalanobis distance. Instead, you should use matrix multiplication 
         for the product of the inverse covariance matrix and the difference between the point and mean
        """
        mahalanobis_term = torch.matmul(embed - mean_k, inv_cov_k)
        """
        Stacking Distances: It seems like you are stacking the distances for each class along the last dimension. If you want
         to get a single Mahalanobis distance for each test point, you should probably use torch.cat instead of torch.stack
        """
        dists_k = torch.sum(mahalanobis_term * (embed - mean_k), dim=1, keepdim=True)  # Ensure dists_k is a column vector

        dists.append(dists_k)

    return torch.cat(dists, dim=1)    

#=------------ MAHALANOBIS METRIC ___________________
def get_mahalnobis_distance_old(embed,ttl_labels,ttl_embed):
    
    classes=torch.unique(ttl_labels)
    
    dists=[]
    
    for k in classes:
        
        embed_k=ttl_embed[ttl_labels==k]
        mean_k=torch.mean(embed_k,0)
        cov_k=torch.cov(embed_k.T)
        
        #print(k)
        #print(cov_k.shape)
        
        inv_cov_k=torch.linalg.inv(cov_k)
        
        dists_k=torch.sum(torch.matmul(embed,inv_cov_k)*embed,1)
        
        #print(dists_k.shape,embed.shape)
        
        dists.append(dists_k)
        
    return torch.stack(dists).T

def compute_confscores_mhnb_minority(model, test_loader,ttl_labels,ttl_embeds, outdir, id_flag, class_idx,args):
    total = 0
    if class_idx == -2:
        if id_flag == True:
            # f'confscores_{args.dataset}_id.txt', f'confscores_{args.dataset}_ood.txt'
            outfile = os.path.join(outdir, f'confscores_{args.dataset}_id.txt')
        else:
            outfile = os.path.join(outdir, f'confscores_{args.dataset}_ood.txt')
        f = open(outfile, 'w')
        for data, labels in test_loader:
            scores, embed = model(data.cpu())
            scores=get_mahalanobis_distance(embed,ttl_labels,ttl_embeds)
            confscores, _ = torch.min(scores, 1)
            total += data.size(0)

            for i in range(data.size(0)):
                f.write("{}\n".format(-confscores[i])) # as it is the mhlb score
        
        f.close()    
        #  f'confscores_{args.dataset}_rest_class_{class_idx}.txt', f'confscores_{args.dataset}_class_{class_idx}.txt')
    else:
        outfile1 = os.path.join(outdir, f'confscores_{args.dataset}_class_{class_idx}.txt')  
        outfile2 = os.path.join(outdir,f'confscores_{args.dataset}_rest_class_{class_idx}.txt') 
        minority_class_indices = [i for i, (_, label) in enumerate(test_loader.dataset) if label == class_idx]
        # rest_class_indices = [i for i in range(len(test_loader.dataset)) if i not in [class_idx]]
        rest_class_indices = [i for i, (_, label) in enumerate(test_loader.dataset) if label != class_idx]

        # Create subsets for minority class and rest of the classes
        minority_class_subset = Subset(test_loader.dataset, minority_class_indices)
        rest_class_subset = Subset(test_loader.dataset, rest_class_indices)

        # Create data loaders for minority class and rest of the classes
        minority_class_loader = DataLoader(minority_class_subset, batch_size=test_loader.batch_size)#, shuffle=test_loader.shuffle)
        rest_class_loader = DataLoader(rest_class_subset, batch_size=test_loader.batch_size)#, shuffle=test_loader.shuffle)
        f = open(outfile1, 'w')
        for data, labels in minority_class_loader:
            scores, embed = model(data.cpu())
            scores=get_mahalanobis_distance(embed,ttl_labels,ttl_embeds)

            confscores = scores[:,class_idx]
            total += data.size(0)

            for i in range(data.size(0)):
                f.write("{}\n".format(confscores[i])) # as it is the softmax score
        
        f.close()
           
        f = open(outfile2, 'w')
        for data, labels in rest_class_loader:
            scores, embed = model(data.cpu())
            scores=get_mahalanobis_distance(embed,ttl_labels,ttl_embeds)
            confscores = scores[:,class_idx]
            total += data.size(0)

            for i in range(data.size(0)):
                f.write("{}\n".format(confscores[i])) # as it is the softmax score
        
        f.close()          

def compute_confscores_mhnb_minority_old(model, test_loader,ttl_labels,ttl_embeds, outdir, id_flag, class_idx,args):
    total = 0
    if class_idx == -2:
        if id_flag == True:
            # f'confscores_{args.dataset}_id.txt', f'confscores_{args.dataset}_ood.txt'
            outfile = os.path.join(outdir, f'confscores_{args.dataset}_id.txt')
        else:
            outfile = os.path.join(outdir, f'confscores_{args.dataset}_ood.txt')
        f = open(outfile, 'w')
        for data, labels in test_loader:
            embed = model(data.cpu())
            scores=get_mahalnobis_distance(embed,ttl_labels,ttl_embeds)
            confscores, _ = torch.min(scores, 1)
            total += data.size(0)

            for i in range(data.size(0)):
                f.write("{}\n".format(-confscores[i]))
        
        f.close()    
        #  f'confscores_{args.dataset}_rest_class_{class_idx}.txt', f'confscores_{args.dataset}_class_{class_idx}.txt')
    else:
        outfile1 = os.path.join(outdir, f'confscores_{args.dataset}_class_{class_idx}.txt')  
        outfile2 = os.path.join(outdir,f'confscores_{args.dataset}_rest_class_{class_idx}.txt') 
        minority_class_indices = [i for i, (_, label) in enumerate(test_loader.dataset) if label == class_idx]
        # rest_class_indices = [i for i in range(len(test_loader.dataset)) if i not in [class_idx]]
        rest_class_indices = [i for i, (_, label) in enumerate(test_loader.dataset) if label != class_idx]

        # Create subsets for minority class and rest of the classes
        minority_class_subset = Subset(test_loader.dataset, minority_class_indices)
        rest_class_subset = Subset(test_loader.dataset, rest_class_indices)

        # Create data loaders for minority class and rest of the classes
        minority_class_loader = DataLoader(minority_class_subset, batch_size=test_loader.batch_size)#, shuffle=test_loader.shuffle)
        rest_class_loader = DataLoader(rest_class_subset, batch_size=test_loader.batch_size)#, shuffle=test_loader.shuffle)
        f = open(outfile1, 'w')
        for data, labels in minority_class_loader:
            embed = model(data.cpu())
            scores=get_mahalnobis_distance(embed,ttl_labels,ttl_embeds)
            confscores = scores[:,class_idx]
            total += data.size(0)

            for i in range(data.size(0)):
                f.write("{}\n".format(-confscores[i]))
        
        f.close()
           
        f = open(outfile2, 'w')
        for data, labels in rest_class_loader:
            embed = model(data.cpu())
            scores=get_mahalnobis_distance(embed,ttl_labels,ttl_embeds)
            confscores = scores[:,class_idx]
            total += data.size(0)

            for i in range(data.size(0)):
                f.write("{}\n".format(-confscores[i]))
        
        f.close()      
# ----------- softmax metrics ----------------
def compute_confscores_softmax_minority(model, test_loader, outdir, id_flag, class_idx,args):
    total = 0
    if class_idx == -2:
        if id_flag == True:
            # f'confscores_{args.dataset}_id.txt', f'confscores_{args.dataset}_ood.txt'
            outfile = os.path.join(outdir, f'confscores_{args.dataset}_id.txt')
        else:
            outfile = os.path.join(outdir, f'confscores_{args.dataset}_ood.txt')
        f = open(outfile, 'w')
        for data, labels in test_loader:
            scores = model(data.cpu())
            scores = torch.nn.functional.softmax(scores)
            confscores, _ = torch.max(scores, 1)
            total += data.size(0)

            for i in range(data.size(0)):
                f.write("{}\n".format(confscores[i]))
        
        f.close()    
        #  f'confscores_{args.dataset}_rest_class_{class_idx}.txt', f'confscores_{args.dataset}_class_{class_idx}.txt')
    else:
        outfile1 = os.path.join(outdir, f'confscores_{args.dataset}_class_{class_idx}.txt')  
        outfile2 = os.path.join(outdir,f'confscores_{args.dataset}_rest_class_{class_idx}.txt') 
        minority_class_indices = [i for i, (_, label) in enumerate(test_loader.dataset) if label == class_idx]
        # rest_class_indices = [i for i in range(len(test_loader.dataset)) if i not in [class_idx]]
        rest_class_indices = [i for i, (_, label) in enumerate(test_loader.dataset) if label != class_idx]

        # Create subsets for minority class and rest of the classes
        minority_class_subset = Subset(test_loader.dataset, minority_class_indices)
        rest_class_subset = Subset(test_loader.dataset, rest_class_indices)

        # Create data loaders for minority class and rest of the classes
        minority_class_loader = DataLoader(minority_class_subset, batch_size=test_loader.batch_size)#, shuffle=test_loader.shuffle)
        rest_class_loader = DataLoader(rest_class_subset, batch_size=test_loader.batch_size)#, shuffle=test_loader.shuffle)
        f = open(outfile1, 'w')
        for data, labels in minority_class_loader:
            scores = model(data.cpu())
            scores = torch.nn.functional.softmax(scores)
            confscores = scores[:,class_idx]
            total += data.size(0)

            for i in range(data.size(0)):
                f.write("{}\n".format(confscores[i]))
        
        f.close()
           
        f = open(outfile2, 'w')
        for data, labels in rest_class_loader:
            scores = model(data.cpu())
            scores = torch.nn.functional.softmax(scores)
            confscores = scores[:,class_idx]
            total += data.size(0)

            for i in range(data.size(0)):
                f.write("{}\n".format(confscores[i]))
        
        f.close()    
# ----------- theirs metrics ----------------
def compute_confscores_theirs_minority(model, test_loader, outdir, id_flag, class_idx,args):
    total = 0
    if class_idx == -2:
        if id_flag == True:
            # f'confscores_{args.dataset}_id.txt', f'confscores_{args.dataset}_ood.txt'
            outfile = os.path.join(outdir, f'confscores_{args.dataset}_id.txt')
        else:
            outfile = os.path.join(outdir, f'confscores_{args.dataset}_ood.txt')
        f = open(outfile, 'w')
        for data, labels in test_loader:
            dists = model(data.cpu())
            confscores, _ = torch.min(dists, dim=1)
            total += data.size(0)

            for i in range(data.size(0)):
                f.write("{}\n".format(-confscores[i]))
        
        f.close()    
        #  f'confscores_{args.dataset}_rest_class_{class_idx}.txt', f'confscores_{args.dataset}_class_{class_idx}.txt')
    else:
        outfile1 = os.path.join(outdir, f'confscores_{args.dataset}_class_{class_idx}.txt')  
        outfile2 = os.path.join(outdir,f'confscores_{args.dataset}_rest_class_{class_idx}.txt') 
        minority_class_indices = [i for i, (_, label) in enumerate(test_loader.dataset) if label == class_idx]
        rest_class_indices = [i for i, (_, label) in enumerate(test_loader.dataset) if label != class_idx]

        # Create subsets for minority class and rest of the classes
        minority_class_subset = Subset(test_loader.dataset, minority_class_indices)
        rest_class_subset = Subset(test_loader.dataset, rest_class_indices)

        # Create data loaders for minority class and rest of the classes
        minority_class_loader = DataLoader(minority_class_subset, batch_size=test_loader.batch_size)#, shuffle=test_loader.shuffle)
        rest_class_loader = DataLoader(rest_class_subset, batch_size=test_loader.batch_size)#, shuffle=test_loader.shuffle)
        f = open(outfile1, 'w')
        for data, labels in minority_class_loader:
            dists = model(data.cpu())
            # confscores, _ = torch.min(dists, dim=1)
            confscores = dists[:,class_idx]
            total += data.size(0)

            for i in range(data.size(0)):
                f.write("{}\n".format(-confscores[i]))
        
        f.close()
           
        f = open(outfile2, 'w')
        for data, labels in rest_class_loader:
            dists = model(data.cpu())
            # confscores, _ = torch.min(dists, dim=1)
            confscores = dists[:,class_idx]
            total += data.size(0)

            for i in range(data.size(0)):
                f.write("{}\n".format(-confscores[i]))
        
        f.close()
# -------------------    # ------------------- # ------------------- # ------------------- # ---------------------    
#--------# -------------------  for ours -------------------# ------------------- # ------------------- # --------
# -------------------    # ------------------- # ------------------- # ------------------- # --------------------- 

def compute_confscores_ours(model, test_loader, outdir, id_flag):
    total = 0
    if id_flag == True:
        outfile = os.path.join(outdir, 'confscores_id.txt')
    else:
        outfile = os.path.join(outdir, 'confscores_ood.txt')

    f = open(outfile, 'w')
    
    for data, _ in test_loader:
        dists,_,_ = model(data.cpu())
        confscores, _ = torch.min(dists, dim=1)
        total += data.size(0)

        for i in range(data.size(0)):
            f.write("{}\n".format(-confscores[i]))
    
    f.close()

def get_auroc_curve(indir):
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
    # at first threshld all samples are positives ---- every thing is marked as positve 
    #tp[0] (ID), fp[0] (OOD) = num_k, num_n
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
    results[mtype] = -np.trapz(1. - fpr, tpr) # for  positive 
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
    
    # DTACC
    mtype = 'DTACC'
    results[mtype] = .5 * (tp/tp[0] + 1. - fp/fp[0]).max() # max oaverage of recall of positive and negative 
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
    
    # AUIN
    mtype = 'AUIN'
    denom = tp + fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]]) 
    pin = np.concatenate([[.5], tp/denom, [0.]]) # precision of positive class  
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])  
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
    
    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0] - tp + fp[0] - fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]]) # precision of negative class  
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
    
# -------------------    # ------------------- # ------------------- # ------------------- # ---------------------    
# -------------------    # ------------------- # ------------------- # ------------------- # ---------------------    
#--------# -------------------  for ours  minority -------------------# ------------------- # ------------------- # --------
# -------------------    # ------------------- # ------------------- # ------------------- # --------------------- 
# # -------------------    # ------------------- # ------------------- # ------------------- # -------------------    
# # -------------------    # ------------------- # ------------------- # ------------------- # ------------------- 
#            
def compute_confscores_ours_minority(model, test_loader, outdir, id_flag, class_idx,args):
    total = 0
    if class_idx == -2:
        if id_flag == True:
            # f'confscores_{args.dataset}_id.txt', f'confscores_{args.dataset}_ood.txt'
            outfile = os.path.join(outdir, f'confscores_{args.dataset}_id.txt')
        else:
            outfile = os.path.join(outdir, f'confscores_{args.dataset}_ood.txt')
        f = open(outfile, 'w')
        for data, labels in test_loader:
            dists, _, _ = model(data.cpu())
            confscores, _ = torch.min(dists, dim=1)
            total += data.size(0)

            for i in range(data.size(0)):
                f.write("{}\n".format(-confscores[i]))
        
        f.close()    
        #  f'confscores_{args.dataset}_rest_class_{class_idx}.txt', f'confscores_{args.dataset}_class_{class_idx}.txt')
    else:
        outfile1 = os.path.join(outdir, f'confscores_{args.dataset}_class_{class_idx}.txt')  
        outfile2 = os.path.join(outdir,f'confscores_{args.dataset}_rest_class_{class_idx}.txt') 
        minority_class_indices = [i for i, (_, label) in enumerate(test_loader.dataset) if label == class_idx]
        # rest_class_indices = [i for i in range(len(test_loader.dataset)) if i not in [class_idx]]
        rest_class_indices  = [i for i, (_, label) in enumerate(test_loader.dataset) if label != class_idx]

        # Create subsets for minority class and rest of the classes
        minority_class_subset = Subset(test_loader.dataset, minority_class_indices)
        rest_class_subset = Subset(test_loader.dataset, rest_class_indices)

        # Create data loaders for minority class and rest of the classes
        minority_class_loader = DataLoader(minority_class_subset, batch_size=test_loader.batch_size)#, shuffle=test_loader.shuffle)
        rest_class_loader = DataLoader(rest_class_subset, batch_size=test_loader.batch_size)#, shuffle=test_loader.shuffle)
        f = open(outfile1, 'w')
        for data, labels in minority_class_loader:
            dists, _, _ = model(data.cpu())
            # confscores, _ = torch.min(dists, dim=1)
            confscores = dists[:,class_idx]
            total += data.size(0)

            for i in range(data.size(0)):
                f.write("{}\n".format(-confscores[i]))
        
        f.close()
           
        f = open(outfile2, 'w')
        for data, labels in rest_class_loader:
            dists, _, _ = model(data.cpu())
            # confscores, _ = torch.min(dists, dim=1)
            confscores= dists[:,class_idx]
            total += data.size(0)

            for i in range(data.size(0)):
                f.write("{}\n".format(-confscores[i]))
        
        f.close()

def get_auroc_curve_ours_minority(indir,known_file, novel_file,class_idx ):
    # known is positive class and novel is negative class-----
    known = np.loadtxt(os.path.join(indir, known_file))#'confscores_id.txt'))#, delimiter='\n')
    novel = np.loadtxt(os.path.join(indir, novel_file))#'confscores_ood.txt'))#, delimiter='\n')
    # compute_metrics_ours_minority(outdir, f'confscores_{args.dataset}_id.txt', f'confscores_{args.dataset}_ood.txt'))
    # compute_metrics_ours_minority(outdir,
    # f'confscores_{args.dataset}_class_{class_idx}.txt', f'confscores_{args.dataset}_rest_class_{class_idx}.txt')

    # In the given code, the known variable represents the confidence scores for the known or in-distribution class 
    # (positive class), and the novel variable represents the confidence scores for the novel or 
    # out-of-distribution class (negative class). ositives are samples from the known or in-distribution class 
    # (class of interest), and they are associated with the known confidence scores.
    # thus, class vs rest in ID samples my positive class is class and negatives are rest there conf = -dists = log(prob)
    # should be more negative thus, probability close to zero
    # hence known = 'confscores_{args.dataset}_class_{class_idx}.txt', and 
    # novel = f'confscores_{args.dataset}_rest_class_{class_idx}.txt'
    if class_idx == -2:
        #the loop processes samples in a way that assumes positive samples (in-distribution)
        #have lower confidence scores, and negative samples (out-of-distribution) 
        #have higher confidence scores.
        known.sort()
        novel.sort()
    else:
        # This will sort the arrays in descending order, ensuring that higher confidence
        #  scores are considered positives during the evaluation.
        known.sort()
        # known = known[::-1]  # This reverses the array, effectively sorting it in descending order
        novel.sort()  
        # novel = novel[::-1]  # This reverses the array, effectively sorting it in descending order
    
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
        # if the confidence score of the out-of-distribution sample (novel[n]) is lower,
        #  it means a False Positive. The counts are updated accordingly.
        #the loop processes samples in a way that assumes positive samples (in-distribution)
        #have lower confidence scores, and negative samples (out-of-distribution) 
        #have higher confidence scores. 
        # novel[n] < known[k] --- 
        # If the confidence score of the in-distribution sample (known[k]) is lower,
        #  it means a True Positive. The counts are updated accordingly.
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


def compute_metrics_ours_minority(dir_name, known_file, novel_file,class_idx, verbose=False):
    tp, fp, tnr_at_tpr85, tnr_at_tpr95 = get_auroc_curve_ours_minority(dir_name, known_file, novel_file, class_idx)
    
    # compute_metrics_ours_minority(outdir, f'confscores_{args.dataset}_id.txt', f'confscores_{args.dataset}_ood.txt'))
    # compute_metrics_ours_minority(outdir,
    # f'confscores_{args.dataset}_class_{class_idx}.txt', f'confscores_{args.dataset}_rest_class_{class_idx}.txt')
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

    """
    If you are interested in a single, aggregate measure of overall performance that considers both 
    positive and negative classes, DTACC might be a suitable choice.
    If you want a more detailed analysis of the model's performance on positive (in-distribution) samples,
    especially across different decision thresholds, AUIN may provide more insights.
    """    
    # In this code, AUIN is calculated by constructing a precision-recall curve (pin) based 
    # on the true positive (tp) and false positive (fp) rates. The area under this curve
    #  is then computed using the negative trapezoidal rule (-np.trapz). This area corresponds
    #  to the AUIN metric, which, as you correctly identified, is analogous to the AUPR 
    # for the positive class.
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


# def print_minority_results_total(minority_result_list):
    
#     TNR85_list = [100.*minority_result['TNR85'] for minority_result in minority_result_list]
#     TNR95_list = [100.*minority_result['TNR95'] for minority_result in minority_result_list]
#     AUROC_list = [100.*minority_result['AUROC'] for minority_result in minority_result_list]
#     DTACC_list = [100.*minority_result['DTACC'] for minority_result in minority_result_list]
#     AUIN_list = [100.*minority_result['AUIN'] for minority_result in minority_result_list]
#     AUOUT_list = [100.*minority_result['AUOUT'] for minority_result in minority_result_list]

#     for mtype in ['TNR85', 'TNR95', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']:
#         print(' {mtype:15s}'.format(mtype=mtype), end='')
#     print('\n{mean:6.2f} ({std:6.3f})'.format(mean=np.mean(TNR85_list), std=np.std(TNR85_list)), end='')
#     print(' {mean:6.2f} ({std:6.3f})'.format(mean=np.mean(TNR95_list), std=np.std(TNR95_list)), end='')
#     print(' {mean:6.2f} ({std:6.3f})'.format(mean=np.mean(AUROC_list), std=np.std(AUROC_list)), end='')
#     print(' {mean:6.2f} ({std:6.3f})'.format(mean=np.mean(DTACC_list), std=np.std(DTACC_list)), end='')
#     print(' {mean:6.2f} ({std:6.3f})'.format(mean=np.mean(AUIN_list), std=np.std(AUIN_list)), end='')
#     print(' {mean:6.2f} ({std:6.3f})\n'.format(mean=np.mean(AUOUT_list), std=np.std(AUOUT_list)), end='')
#     print('')            
