import torch
import numpy as np

# main attack
class RamBoAtt():
    def __init__(self,
                model,
                m_block = 16,  # 1 for CIFAR-10
                lamda=2,       # 1.2 for CIFAR-10
                w=1,
                len_T =1000,   # 500 for CIFAR-10
                delta = 1e-1,  # 1e-2 for CIFAR-10
                seed = None,
                targeted=True):

        self.model = model
        self.m_block = m_block # number of block
        self.lamda = lamda 
        self.w = w
        self.len_T = len_T
        self.delta = delta
        self.seed = seed
        self.targeted = targeted

    def convert_idx_ImgN(self,n, x):
        # c1 = Channel = C
        # c2 = Width = W
        # c3 = Height = H
        # c1 x c2 x c3 = C x W x H
        c1 = n // (x.shape[2] * x.shape[3])
        c2 = (n - c1 * x.shape[2] * x.shape[3])// x.shape[3]
        c3 = n - c1 * x.shape[2] * x.shape[3] - c2 * x.shape[3]
        return c1,c2,c3

    def BlockDescent(self,ori_img, ori_label, target_img, target_label,eps,max_qry):
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
            while i < (n_dims-self.m_block):
                best_adv_temp = best_adv.clone()
                for k in range(self.m_block):
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
                    w1 = torch.min(torch.tensor([self.w,c2]))
                    w2 = torch.min(torch.tensor([self.w,wi-c2])) + 1
                    h1 = torch.min(torch.tensor([self.w,c3]))
                    h2 = torch.min(torch.tensor([self.w,he-c3])) + 1
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
                    if nqry % self.len_T == 0:
                        DR = np.mean(D[nqry - self.len_T:nqry])
                        if ((DL-DR) < self.delta): #
                            terminate = True
                            print('\nBreak due to slow convergence!\n')
                            break
                        else:
                            DL = DR
                if nqry<max_qry:
                    i += self.m_block
                else:
                    terminate = True
                    print('\nBreak due to exceeding query limit!\n')
                    break

            eps /= self.lamda
            
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