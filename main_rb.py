import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0";
#from pudb import set_trace
import numpy as np
import torch
from torchvision import models

import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm


from utils_rb import *
from ramboattack import RamBoAtt
from HSJA_rb import HSJA
from SignOPT_rb import OPT_attack_sign_SGD

import argparse

# e.g code (README)
# python main_se_l0.py --dataset imagenet --arch resnet50 --attack_setting targeted --n_start 0 --n_end 100 --query_limit 20000 --mu 0.001
# model_path = './models/pgd_adversarial_training_original.pth'
# @profile => just use when checking memory using by each line of code

def get_args():

    # ========== M0: args input data ===========    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--dataset", default='imagenet', type=str, help="cifar10, cifar100 or imagenet.")
    parser.add_argument("--arch", default='resnet50', type=str, help="cifar10_gpu or resnet50")
    parser.add_argument("--model_path", default=None, type=str, help="Path to a pretrained model")
    #parser.add_argument("--data_path", default=None, type=str, help="Path to a dataset")
    parser.add_argument("--output_dir", default='results', type=str, help="Dir to an output file")
    parser.add_argument("--eval_set", default='balance', type=str, help="balance (imagenet & CIFAR10), easyset (imagenet & CIFAR10),hardset (imagenet) ,hardset_A (CIFAR10),hardset_B (CIFAR10),hardset_D (CIFAR10)")

    # e.g. "args.unnorm" should be "True" for the pretrained model "cifar10_gpu.pt" used in Sign-OPT or OPT-attack because this mode does not require "normalized" data.
    parser.add_argument("--unnorm", action='store_true', help="if a pretrained model requires normalized data, args.unnorm should be False, default = False") 
    
    parser.add_argument("--targeted", action='store_true', help="True or False, default = False (untargeted).")
    parser.add_argument("--n_start", default=0, type=int, help="first sample of eval_set")
    parser.add_argument("--n_end", default=100, type=int, help="last sample of eval_set")
    parser.add_argument("--query_limit", default=50000, type=int, help="query budget for the attack")
    #parser.add_argument("--lambda", default=2.0, type=float, help="control perturbation magnitude")
    #parser.add_argument("--m", default = 16, type=float, help="number of block")
    parser.add_argument("--rambo_type", default='RBH', type=str, help="RBH = RamBo(HSJA) or RBS = RamBo(SOPT)")
    #parser.add_argument("--defense", default='standard', type=str, help="Standard (undefended) or AT, Distill or Region_based (defend mechanism)")
    args = parser.parse_args()

    return args

def main(args):

    # ======================================================================================================================================================
    # ========== M1: Load data =========== 
    # ======================================================================================================================================================
    batch_size = 1
    testloader, testset = load_data(args.dataset,batch_size=batch_size)
    print('=> M1: Load data successfully !!!')

    # ======================================================================================================================================================
    # ========== M2: Load and draft model =========== 
    # ======================================================================================================================================================

    net = load_model(args.arch,args.model_path)
    model_rb = PretrainedModel(net,args.dataset,args.unnorm)
    
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'imagenet':
        num_classes = 1000
    bounds = [0,1]
    model_ex = PytorchModel_ex(net, bounds, num_classes,args.dataset,args.unnorm)
    print('=> M2: Load model successfully !!!')

    # ======================================================================================================================================================
    # ========== M3: Get evaluation set =========== 
    # ======================================================================================================================================================

    seed = 999
    ID_set = get_evalset(model_rb,args.dataset, args.arch,testset,seed,args.targeted,args.eval_set)
    print('=> M3: Generate eval_set successfully !!!')

    # ======================================================================================================================================================
    # ========== M4: attack setup =========== 
    # ======================================================================================================================================================
    
    # ============ module 1 ===============
    if args.rambo_type == 'RBH':
        constraint='l2'
        num_iterations=150
        gamma=1.0
        stepsize_search='geometric_progression'
        max_num_evals = 1e4
        init_num_evals=100
        verbose=True
        if args.dataset == 'cifar10':
            delta = 1e-2
            len_T = 500
        elif args.dataset == 'imagenet':
            delta = 1
            len_T = 2000       
        module_1 = HSJA(model_ex,constraint,num_iterations,gamma,stepsize_search,max_num_evals,init_num_evals, verbose,delta,len_T)
    elif args.rambo_type == 'RBS':
        k=200
        test_dataset=testset
        if args.dataset == 'cifar10':
            delta = 1e-2
            len_T = 2
        elif args.dataset == 'imagenet':
            delta = 1
            len_T = 4
        #module_1 = OPT_attack_sign_SGD(model_ex,k,delta,len_T,testset)
        module_1 = OPT_attack_sign_SGD(model_rb,k,delta,len_T,testset)

    # ============ module 2 ===============
    if args.dataset == 'cifar10':
        m_block = 1   
        lamda = 1.2     
        w=1
        len_T = 500 
        delta = 1e-2 

    elif args.dataset == 'imagenet':
        m_block = 16   
        lamda = 2      
        w = 1
        len_T = 1000   
        delta = 1e-1 
    module_2 = RamBoAtt(model_rb,m_block,lamda,w,len_T,delta,seed,args.targeted)

    # ============ module 3 ===============
    if args.dataset == 'cifar10':
        delta = 1e-2
        len_T = 2
    elif args.dataset == 'imagenet':
        delta = 1
        len_T = 4
    k=200
    #module_3 = OPT_attack_sign_SGD(model_ex,k,delta,len_T,testset)
    module_3 = OPT_attack_sign_SGD(model_rb,k,delta,len_T,testset)

    print('=> M4: Attack setup Done !!!')

    # ======================================================================================================================================================
    # ========== M5: output setup ===========
    # ======================================================================================================================================================

    n_point = 100 # a number of datapoint over query limit
    if args.targeted:
        file_name = 'RamBoAttack_'+args.rambo_type +'_'+args.arch+'_'+args.dataset+'_'+args.eval_set+'_Targeted_Fr'+ str(args.n_start)+'_To'+str(args.n_end)+'.csv'
        output_path = os.path.join(args.output_dir,file_name)
    else:
        file_name = 'RamBoAttack_'+args.rambo_type +'_'+args.arch+'_'+args.dataset+'_'+args.eval_set+'_Untargeted_Fr'+ str(args.n_start)+'_To'+str(args.n_end)+'.csv'
        output_path = os.path.join(args.output_dir,file_name)

    if args.targeted:
        head = ['#','ocla','o_ID','tcla','t_ID','alabel']

    else:
        head = ['#','ocla','o_ID','alabel']

    for k in range(n_point):
        head.append('q'+str(k))

    print('=> M5: Output setup Done !!!')

    # ======================================================================================================================================================
    # ========== M6: Evaluation attack =========== 
    # ======================================================================================================================================================

    print('=> M6: Evaluation in progress ...')
    
    for i in tqdm(range(args.n_start,args.n_end),desc='Sample'):                          # ori_class - 10
        print(i)
    
        D = np.zeros(args.query_limit+2000)
        nquery = 0
        o = ID_set[i,1] #oID

        # 0. select original image
        oimg, olabel = testset[o]
        oimg = torch.unsqueeze(oimg, 0).cuda()

        # 1. select starting image
        if args.targeted:
            t = ID_set[i,3] #tID, 3 is index acrross dataset - 4 is sample index in a class (not accross dataset)
            tlabel = ID_set[i,2]
            #timg, tlabel = testset[t]
            timg, _ = testset[t]
            timg = torch.unsqueeze(timg, 0).cuda()
            y_targ = np.array([tlabel])
        else:
            tlabel = None
            y_targ = np.array([olabel])


        # 2. Run attack
        # =========== module 1 ===========

        if args.rambo_type == 'RBH':
            model_ex = PytorchModel_ex(net, bounds, num_classes,args.dataset,args.unnorm)
            module_1 = HSJA(model_ex,constraint,num_iterations,gamma,stepsize_search,max_num_evals,init_num_evals, verbose,delta,len_T)
            if args.targeted:
                adv, nqry, Dt = module_1.hsja(oimg.cpu().numpy(), y_targ, timg.cpu().numpy(),args.targeted)
            else:
                timg = None
                adv, nqry, Dt = module_1.hsja(oimg.cpu().numpy(), y_targ, timg,args.targeted)
            timg = torch.unsqueeze(torch.from_numpy(adv).float(), 0).cuda()
            print('Module 1: Finished HSJA')
        
        elif args.rambo_type == 'RBS':
            alpha = 0.2
            beta = 0.001
            iterations = 5000
            query_limit = args.query_limit
            distortion = None
            stopping = 0.0001
            auto_terminate=True

            if args.targeted:
                adv, nqry, Dt = module_1.attack_targeted(oimg, olabel, timg, tlabel, alpha, beta, iterations, query_limit, distortion, seed, stopping, auto_terminate)
            else:
                timg = None
                adv, nqry, Dt = module_1.attack_untargeted(oimg, olabel, timg ,alpha, beta, iterations, query_limit, distortion,seed, stopping, auto_terminate)
            timg = adv.cuda()
            print('Module 1: Finished SignOPT')

        D[nquery:nquery + nqry] = Dt
        nquery += nqry

        # =========== module 2 ===========
        if nquery<args.query_limit:
            max_query = args.query_limit - nquery
            if args.dataset == 'cifar10':
                pi = 100
            elif args.dataset == 'imagenet':
                pi = 50
            eps = np.percentile(torch.abs(timg - oimg).cpu().numpy(), pi)
            adv, nqry, Dt = module_2.BlockDescent(oimg,olabel,timg,tlabel,eps,max_query)

            print('Module 2: Finished BlockDescent')

            D[nquery:nquery + nqry] = Dt
            nquery += nqry

        # =========== module 3 ===========
        if nquery<args.query_limit:
            alpha = 0.2
            beta = 0.001
            iterations = 5000
            query_limit = args.query_limit - nquery
            distortion = None
            stopping = 0.0001
            auto_terminate=False

            timg = adv
            if args.targeted:
                adv, nqry, Dt = module_3.attack_targeted(oimg, olabel, timg, tlabel, alpha, beta, iterations, query_limit, distortion, seed, stopping, auto_terminate)
            else:
                adv, nqry, Dt = module_3.attack_untargeted(oimg, olabel, timg ,alpha, beta, iterations, query_limit, distortion,seed, stopping, auto_terminate)
            
            print('Module 3: Finished SignOPT')

            D[nquery:nquery + nqry] = Dt
            nquery += nqry

        # 3. write it out
        alabel = model_rb.predict_label(adv)

        if args.targeted:
            key_info = [i,olabel,o,tlabel,t,alabel.item()]
        else:
            key_info = [i,olabel,o,alabel.item()]

        export_pd_csv(D[:nquery],head,key_info,output_path,n_point,args.query_limit)

if __name__ == "__main__":

    args = get_args()
    main(args)
