import torch
import numpy as np
from utils_rb import *
from HSJA_rb import HSJA
from SignOPT_rb import OPT_attack_sign_SGD

# main attack
class RamBoAtt():
    def __init__(self,
                model,
                model_ex,
                testset,
                #m_block = 16,  # 1 for CIFAR-10
                #lamda=2,       # 1.2 for CIFAR-10
                #w=1,
                #len_T =1000,   # 500 for CIFAR-10
                #delta = 1e-1,  # 1e-2 for CIFAR-10
                seed = None,
                targeted=True,
                dataset='cifar10'):

        self.model = model
        self.model_ex = model_ex
        self.testset = testset
        #self.m_block = m_block # number of block
        #self.lamda = lamda 
        #self.w = w
        #self.len_T = len_T
        #self.delta = delta
        self.seed = seed
        self.targeted = targeted
        self.dataset=dataset

    def convert_idx_ImgN(self,n, x):
        # c1 = Channel = C
        # c2 = Width = W
        # c3 = Height = H
        # c1 x c2 x c3 = C x W x H
        c1 = n // (x.shape[2] * x.shape[3])
        c2 = (n - c1 * x.shape[2] * x.shape[3])// x.shape[3]
        c3 = n - c1 * x.shape[2] * x.shape[3] - c2 * x.shape[3]
        return c1,c2,c3

    def BlockDescent(self,ori_img, ori_label, target_img, target_label,eps,max_qry,
                     delta,m_block,lamda,w,len_T):
        # With view
        best_adv = target_img.clone()
        D = np.zeros(max_qry+1000)
        wi = ori_img.shape[2]
        he = ori_img.shape[3]
        n_dims = ori_img.view(1, -1).size(1) # ori_img.nelement() = C x W x H  = ori_img.nelement()
        DL = np.inf
        DR = 0
        cnt = 0
        prev_qry = 0
        nqry = 0
        if self.seed != None:
            torch.manual_seed(self.seed)
            
        terminate = False

        while not(terminate):
            idx = torch.randperm(n_dims)
            i = 0
            while i < (n_dims-m_block):
                best_adv_temp = best_adv.clone()
                for k in range(m_block):
                    c1,c2,c3 = self.convert_idx_ImgN(idx[i+k], ori_img)
                    
                    '''
                    if (self.w<=c2) & (c2<=wi-self.w) & (self.w<=c3) & (c3<=he-self.w):
                        mask_sign = torch.sign(ori_img[0,c1, c2-self.w:c2+w+1, c3-self.w:c3+self.w+1]-best_adv[0,c1,c2-self.w:c2+self.w+1,c3-w:c3+w+1])
                        best_adv_temp[0,c1,c2-w:c2+w+1,c3-w:c3+w+1] = (best_adv[0,c1,c2-w:c2+w+1,c3-w:c3+w+1] + eps * mask_sign).clamp(0,1)
                    else:
                        w1 = torch.min(torch.tensor([self.w,c2]))
                        w2 = torch.min(torch.tensor([self.w,wi-c2])) + 1
                        h1 = torch.min(torch.tensor([self.w,c3]))
                        h2 = torch.min(torch.tensor([self.w,he-c3])) + 1
                        mask_sign =  torch.sign(ori_img[0, c1, c2-w1:c2+w2, c3-h1:c3+h2]-best_adv[0, c1, c2-w1:c2+w2, c3-h1:c3+h2])
                        best_adv_temp[0, c1, c2-w1:c2+w2, c3-h1:c3+h2] = (best_adv[0, c1, c2-w1:c2+w2, c3-h1:c3+h2] + eps * mask_sign).clamp(0,1)
                    '''
                    w1 = torch.min(torch.tensor([w,c2]))
                    w2 = torch.min(torch.tensor([w,wi-c2])) + 1
                    h1 = torch.min(torch.tensor([w,c3]))
                    h2 = torch.min(torch.tensor([w,he-c3])) + 1
                    mask_sign =  torch.sign(ori_img[0, c1, c2-w1:c2+w2, c3-h1:c3+h2]-best_adv[0, c1, c2-w1:c2+w2, c3-h1:c3+h2])
                    best_adv_temp[0, c1, c2-w1:c2+w2, c3-h1:c3+h2] = (best_adv[0, c1, c2-w1:c2+w2, c3-h1:c3+h2] + eps * mask_sign).clamp(0,1)
                
                if torch.norm(best_adv_temp - ori_img)< torch.norm(best_adv - ori_img):
                    next_pert_lbl = self.model.predict_label(best_adv_temp)
                    if self.targeted == True:
                        if (next_pert_lbl==target_label):
                            best_adv = best_adv_temp.clone()
                    else:
                        if (next_pert_lbl!=ori_label):
                            best_adv = best_adv_temp.clone()
                    if (nqry%1000)==0:
                        print('Qry#',nqry,'; l2 distance =', torch.norm(best_adv - ori_img).item(),'; adv label:',
                              self.model.predict_label(best_adv).item())
                    D[nqry] = torch.norm(best_adv - ori_img)
                    nqry += 1 
                    
                    # control auto terminate
                    if nqry % len_T == 0:
                        DR = np.mean(D[nqry - len_T:nqry])
                        if ((DL-DR) < delta): #
                            terminate = True
                            print('\nBreak due to slow convergence!\n')
                            break
                        else:
                            DL = DR
                if nqry<max_qry:
                    i += m_block
                else:
                    terminate = True
                    print('\nBreak due to exceeding query limit!\n')
                    break

            eps /= lamda
            
            # engineering to terminate if looping infinitely without any improvement!
            if prev_qry == nqry:
                if cnt ==2:
                    print('Break due to loop infinitely!')
                    break
                else:
                    cnt += 1
            else: 
                cnt = 0
                prev_qry = nqry    

        return best_adv, nqry, D[:nqry]

    def hybrid_attack(self,oimg,olabel,timg,tlabel,query_limit=50000,attack_mode="RBH"):

        # =========== module 1 ===========
        D = np.zeros(query_limit+2000)
        if attack_mode=="RBH":
            if self.targeted:
                y_targ = np.array([tlabel])
            else:
                y_targ = np.array([olabel])

            if self.dataset == 'cifar10':
                delta = 1e-2
                len_T = 500
            elif self.dataset == 'imagenet':
                delta = 1
                len_T = 2000  
            # ========================
            constraint='l2'
            num_iterations=150
            gamma=1.0
            stepsize_search='geometric_progression'
            max_num_evals = 1e4
            init_num_evals=100
            verbose=True
            auto_terminate=True

            module_1 = HSJA(self.model_ex,constraint,num_iterations,gamma,stepsize_search,max_num_evals,init_num_evals, verbose,delta,len_T)
            
            if self.targeted:
                adv, nqry, Dt = module_1.hsja(oimg.cpu().numpy(), y_targ, timg.cpu().numpy(),self.targeted,query_limit,auto_terminate)
            else:
                timg = None
                adv, nqry, Dt = module_1.hsja(oimg.cpu().numpy(), y_targ, timg,self.targeted)
            timg = torch.unsqueeze(torch.from_numpy(adv).float(), 0).cuda()
                    
            print('Module 1: Finished HSJA\n')

        elif attack_mode=='RBS':
            k=200
            if self.dataset == 'cifar10':
                delta = 1e-2
                len_T = 2
            elif self.dataset == 'imagenet':
                delta = 1
                len_T = 4
            auto_terminate=True
            module_1 = OPT_attack_sign_SGD(self.model,k,delta,len_T,self.testset)

            alpha = 0.2
            beta = 0.001
            iterations = 5000
            distortion = None
            stopping = 0.0001
            # ========================

            if self.targeted:
                adv, nqry, Dt = module_1.attack_targeted(oimg, olabel, timg, tlabel, alpha, beta, iterations, query_limit, distortion, self.seed, stopping, auto_terminate)
            else:
                timg = None
                adv, nqry, Dt = module_1.attack_untargeted(oimg, olabel, timg ,alpha, beta, iterations, query_limit, distortion,self.seed, stopping, auto_terminate)

            timg = adv.cuda()
            print('Module 1: Finished SignOPT\n')

        D[:nqry] = Dt
        nquery = nqry

        # =========== module 2 ===========
        if nquery<query_limit:
            max_query = query_limit - nquery
            if self.dataset == 'cifar10':
                pi = 100
                delta = 1e-2
                m_block = 1   
                lamda = 1.2     
                w=1
                len_T = 500 
            elif self.dataset == 'imagenet':
                pi = 50
                delta = 1e-1
                m_block = 16
                lamda = 2     
                w=1
                len_T = 1000 

            eps = np.percentile(torch.abs(timg - oimg).cpu().numpy(), pi)
            adv, nqry, Dt = self.BlockDescent(oimg,olabel,timg,tlabel,eps,max_query,
                                              delta,m_block,lamda,w,len_T)

            print('Module 2: Finished BlockDescent')

            D[nquery:nquery + nqry] = Dt
            nquery += nqry

        # =========== module 3 ===========
        if nquery<query_limit:
            alpha = 0.2
            beta = 0.001
            iterations = 5000
            query_limit = query_limit - nquery
            distortion = None
            stopping = 0.0001
            auto_terminate=False
            if self.dataset == 'cifar10':
                delta = 1e-2
                len_T = 2
            elif self.dataset == 'imagenet':
                delta = 1
                len_T = 4
            k=200
            module_3 = OPT_attack_sign_SGD(self.model,k,delta,len_T,self.testset)
            
            timg = adv
            if self.targeted:
                adv, nqry, Dt = module_3.attack_targeted(oimg, olabel, timg, tlabel, alpha, beta, iterations, query_limit, distortion, self.seed, stopping, auto_terminate)
            else:
                adv, nqry, Dt = module_3.attack_untargeted(oimg, olabel, timg ,alpha, beta, iterations, query_limit, distortion,self.seed, stopping, auto_terminate)
                    
            print('Module 3: Finished SignOPT')

            D[nquery:nquery + nqry] = Dt

        return adv, nqry, D