import time
import numpy as np 
from numpy import linalg as LA
import torch
import scipy.spatial
from scipy.linalg import qr
#from qpsolvers import solve_qp
import random

start_learning_rate = 1.0
MAX_ITER = 1000

class OPT_attack_sign_SGD(object):
    def __init__(self, model, k=200, delta=1e-2,len_T=2,testset=None):
        self.model = model
        self.k = k
        self.delta = delta
        self.len_T = len_T
        self.test_dataset = testset 
        self.log = torch.ones(MAX_ITER,2)

    def get_log(self):
        return self.log
    
    def attack_untargeted(self, x0, y0, xt, alpha = 0.2, beta = 0.001, iterations = 1000, query_limit=20000,
                          distortion=None, seed=None,stopping=0.0001,auto_terminate=True):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            test_dataset: set of training data
            (x0, y0): original image
        """


        model = self.model
        #y0 = y0[0]
        query_count = 0
        ls_total = 0
        
        if (model.predict_label(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return x0
        
        if seed is not None:
            np.random.seed(seed)

        #===========================
        DL = np.inf
        DR = 0
        #m = 1000
        D = np.zeros(query_limit + 1000)
        nq = 0
        #===========================

        # Calculate a good starting point.
        num_directions = 100
        #print("Searching for the initial direction on %d random directions: " % (num_directions))
        timestart = time.time()
        '''
        for i in range(num_directions):
            query_count += 1
            theta = np.random.randn(*x0.shape)
            if model.predict_label(x0+torch.tensor(theta, dtype=torch.float).cuda())!=y0:
                initial_lbd = LA.norm(theta)
                theta /= initial_lbd
                lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    print("--------> Found distortion %.4f" % g_theta)

        if g_theta == float('inf'):
            num_directions = 100
            best_theta, g_theta = None, float('inf')
            print("Searching for the initial direction on %d random directions: " % (num_directions))
            timestart = time.time()
            for i in range(num_directions):
                query_count += 1
                theta = np.random.randn(*x0.shape)
                if model.predict_label(x0+torch.tensor(theta, dtype=torch.float).cuda())!=y0:
                    initial_lbd = LA.norm(theta)
                    theta /= initial_lbd
                    lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                    query_count += count
                    if lbd < g_theta:
                        best_theta, g_theta = theta, lbd
                        print("--------> Found distortion %.4f" % g_theta)
        '''

        initial_require = auto_terminate # both these condition is only used for the first module SignOPT-untarget(before RamBo) 

        if initial_require:
            best_theta, g_theta = None, float('inf')
            for i in range(num_directions):
                query_count += 1
                theta = np.random.randn(*x0.shape)
                if model.predict_label(x0+torch.tensor(theta, dtype=torch.float).cuda())!=y0:
                    initial_lbd = np.linalg.norm(theta)
                    theta /= initial_lbd
                    lbd, count = self.fine_grained_binary_search(model, x0, y0,  theta, initial_lbd, g_theta)
                    query_count += count
                    if lbd < g_theta:
                        best_theta, g_theta = theta, lbd
                        print("--------> Found distortion %.4f at direction %d" % (g_theta,i))
                #=================================
                D[nq:query_count] = g_theta
                nq = query_count
                #=================================
                #print('i: {}, Query: {}' .format(i,query_count))
                #print(f'i: {i}, Query: {query_count}')
                #print('i: ',i,', Query: ',query_count)
                #print('i: %d, Query: %d' %(i,query_count))
        else:
            xi = xt.clone()
            theta = xi.cpu().numpy() - x0.cpu().numpy()
            lbd = np.linalg.norm(theta)
            best_theta, g_theta = theta, lbd
        
        timeend = time.time()        
        if g_theta == float('inf'):    
            print("Couldn't find valid initial, failed")
            return x0 
        print("==========> Found best distortion %.4f in %.4f seconds "
              "using %d queries" % (g_theta, timeend-timestart, query_count))
    
        #self.log[0][0], self.log[0][1] = g_theta, query_count
        
        # Begin Gradient Descent.
        timestart = time.time()
        xg, gg = best_theta, g_theta
        vg = np.zeros_like(xg)
        learning_rate = start_learning_rate
        prev_obj = 100000
        distortions = [gg]
        for i in range(iterations):
            sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta)
            
            # Line search
            ls_count = 0
            min_theta = xg
            min_g2 = gg
            min_vg = vg
            for _ in range(15):
                '''
                if momentum > 0:
#                     # Nesterov
#                     vg_prev = vg
#                     new_vg = momentum*vg - alpha*sign_gradient
#                     new_theta = xg + vg*(1 + momentum) - vg_prev*momentum
                    new_vg = momentum*vg - alpha*sign_gradient
                    new_theta = xg + new_vg
                else:
                    new_theta = xg - alpha * sign_gradient
                '''
                new_theta = xg - alpha * sign_gradient
                new_theta /= LA.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local(
                    model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                ls_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                    #if momentum > 0:
                    #    min_vg = new_vg
                else:
                    break

            if min_g2 >= gg:
                for _ in range(15):
                    alpha = alpha * 0.25
                    '''
                    if momentum > 0:
#                         # Nesterov
#                         vg_prev = vg
#                         new_vg = momentum*vg - alpha*sign_gradient
#                         new_theta = xg + vg*(1 + momentum) - vg_prev*momentum
                        new_vg = momentum*vg - alpha*sign_gradient
                        new_theta = xg + new_vg
                    else:
                        new_theta = xg - alpha * sign_gradient
                    '''
                    new_theta = xg - alpha * sign_gradient
                    new_theta /= LA.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local(
                        model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                    ls_count += count
                    if new_g2 < gg:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        #if momentum > 0:
                        #    min_vg = new_vg
                        break
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving")
                beta = beta*0.1
                if (beta < 1e-8):
                    break
            
            xg, gg = min_theta, min_g2
            vg = min_vg
            
            query_count += (grad_queries + ls_count)
            #=================================
            D[nq:query_count] = gg
            nq = query_count
            #=================================
            ls_total += ls_count
            distortions.append(gg)

            if query_count > query_limit:
               break
            
            if (i+1)%10==0:
                print("Iteration %3d distortion %.4f num_queries %d" % (i+1, gg, query_count))
            #self.log[i+1][0], self.log[i+1][1] = gg, query_count
            #if distortion is not None and gg < distortion:
            #    print("Success: required distortion reached")
            #    break
            if auto_terminate:
                if i%self.len_T==0:
                    DR = gg
                    if (DL-DR) < self.delta:
                        print('\n break due to slow convergence and group dim = 1!\n')
                        break
                    else:
                        DL = DR
#             if gg > prev_obj-stopping:
#                 print("Success: stopping threshold reached")
#                 break            
#             prev_obj = gg
        #target = model.predict_label(x0 + torch.tensor(gg*xg, dtype=torch.float).cuda())
        timeend = time.time()
        #print("\nAdversarial Example Found Successfully: distortion %.4f target"
        #      " %d queries %d \nTime: %.4f seconds" % (gg, target, query_count, timeend-timestart))
        
        #self.log[i+1:,0] = gg
        #self.log[i+1:,1] = query_count
        #print(self.log)
        #print("Distortions: ", distortions)
        #return x0 + torch.tensor(gg*xg, dtype=torch.float).cuda(), gg, query_count, M[:query_count]
        adv_target = self.model.predict_label(x0 + torch.tensor(gg*xg, dtype=torch.float).cuda())
        if (adv_target != y0):
            timeend = time.time()
            print("\nAdversarial Example Found Successfully: distortion %.4f target"
                  " %d queries %d LS queries %d \nTime: %.4f seconds" % (gg, adv_target, query_count, ls_total, timeend-timestart))

            return x0 + torch.tensor(gg*xg, dtype=torch.float).cuda(), query_count, D[:query_count]
        else:
            print("Failed to find targeted adversarial example.")
            return x0, nquery_count, D[:query_count]

    def sign_grad_v1(self, x0, y0, theta, initial_lbd, h=0.001, D=4, target=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        K = self.k
        sign_grad = np.zeros(theta.shape)
        queries = 0
        ### USe orthogonal transform
        #dim = np.prod(sign_grad.shape)
        #H = np.random.randn(dim, K)
        #Q, R = qr(H, mode='economic')
        preds = []
        for iii in range(K):
#             # Code for reduced dimension gradient
#             u = np.random.randn(N_d,N_d)
#             u = u.repeat(D, axis=0).repeat(D, axis=1)
#             u /= LA.norm(u)
#             u = u.reshape([1,1,N,N])
            
            u = np.random.randn(*theta.shape)
            #u = Q[:,iii].reshape(sign_grad.shape)
            u /= LA.norm(u)
            
            sign = 1
            new_theta = theta + h*u
            new_theta /= LA.norm(new_theta)
            
            # Targeted case.
            if (target is not None and 
                self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()) == target):
                sign = -1
                
            # Untargeted case
            preds.append(self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()).item())
            if (target is None and
                self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()) != y0):
                sign = -1
            queries += 1
            sign_grad += u*sign
        
        sign_grad /= K

#         sign_grad_u = sign_grad/LA.norm(sign_grad)
#         new_theta = theta + h*sign_grad_u
#         new_theta /= LA.norm(new_theta)
#         fxph, q1 = self.fine_grained_binary_search_local(self.model, x0, y0, new_theta, initial_lbd=initial_lbd, tol=h/500)
#         delta = (fxph - initial_lbd)/h
#         queries += q1
#         sign_grad *= 0.5*delta       
        
        return sign_grad, queries

    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
         
        if model.predict_label(x0+torch.tensor(lbd*theta, dtype=torch.float).cuda()) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while model.predict_label(x0+torch.tensor(lbd_hi*theta, dtype=torch.float).cuda()) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while model.predict_label(x0+torch.tensor(lbd_lo*theta, dtype=torch.float).cuda()) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).cuda()) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if model.predict_label(x0+torch.tensor(current_best*theta, dtype=torch.float).cuda()) == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd
        
        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).cuda()) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

# =========================================================================================================
# =====================================   Targeted Attack   ===============================================
# =========================================================================================================

    def attack_targeted(self, x0, y0, xt, yt, alpha = 0.2, beta = 0.001, iterations = 5000, query_limit=40000,
                        distortion=None, seed=None, stopping=0.0001,auto_terminate=True):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            test_dataset: set of training data
            (x0, y0): original image
        """
        model = self.model
        #y0 = y0[0]
        print("Targeted attack - Source: {0} and Target: {1} - pred y0: {2} - pred yt: {3}".format(y0, yt,model.predict_label(x0),model.predict_label(xt)))
        
        if (model.predict_label(x0) == yt):
            print("Image already target. No need to attack.")
            return x0, 0.0
        
        if self.test_dataset is None:
            print("Need training dataset for initial theta.")
            return x0, 0.0
        
        if seed is not None:
            np.random.seed(seed)

        #===========================
        DL = np.inf
        DR = 0
        #m = 1000
        D = np.zeros(query_limit + 2000)
        nq = 0
        #===========================

        #num_samples = 100
        best_theta, g_theta = None, float('inf')
        query_count = 0
        ls_total = 0
        #sample_count = 0
        #print("Searching for the initial direction on %d samples: " % (num_samples))
        timestart = time.time()

#         samples = set(random.sample(range(len(self.test_dataset)), num_samples))
#         print(samples)
#         test_dataset = self.test_dataset[samples]

        # Iterate through training dataset. Find best initial point for gradient descent.
        #for i, (xi, yi) in enumerate(self.test_dataset):
        #    yi_pred = model.predict_label(xi.cuda())
        #    query_count += 1
        #    if yi_pred != target:
        #        continue
            
        #    theta = xi.cpu().numpy() - x0.cpu().numpy()
        #    initial_lbd = LA.norm(theta)
        #    theta /= initial_lbd
        #    lbd, count = self.fine_grained_binary_search_targeted(model, x0, y0, target, theta, initial_lbd, g_theta)
        #    query_count += count
        #    if lbd < g_theta:
        #        best_theta, g_theta = theta, lbd
        #        print("--------> Found distortion %.4f" % g_theta)

        #    sample_count += 1
        #    if sample_count >= num_samples:
        #        break

        #    if i > 500:
        #        break

        xi = xt.clone()
        theta = xi.cpu().numpy() - x0.cpu().numpy()
        initial_lbd = np.linalg.norm(theta)
        theta /= initial_lbd
        lbd, count = self.fine_grained_binary_search_targeted(model, x0, y0, yt, theta, initial_lbd, g_theta)
        query_count += count
        #=================================
        best_theta, g_theta = theta, lbd
        print("--------> Found distortion %.4f" % g_theta)

        D[nq:query_count] = g_theta
        nq = query_count
        #=================================

#         xi = initial_xi
#         xi = xi.numpy()
#         theta = xi - x0
#         initial_lbd = LA.norm(theta.flatten(),np.inf)
#         theta /= initial_lbd     # might have problem on the defination of direction
#         lbd, count, lbd_g2 = self.fine_grained_binary_search_local_targeted(model, x0, y0, yt, theta)
#         query_count += count
#         if lbd < g_theta:
#             best_theta, g_theta = theta, lbd
#             print("--------> Found distortion %.4f" % g_theta)        

        #if lbd < g_theta:
        #    best_theta, g_theta = theta, lbd
        #    print("--------> Found distortion %.4f" % g_theta)

        timeend = time.time()
        if g_theta == np.inf:
            return x0, float('inf')
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" %
              (g_theta, timeend-timestart, query_count))
    
        #================================================================================
        # Begin Gradient Descent.
        #================================================================================
        timestart = time.time()
        xg, gg = best_theta, g_theta
        learning_rate = start_learning_rate
        #prev_obj = 100000
        distortions = [gg]
        for i in range(iterations):
            sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta, target=yt)
            
            # Line search
            ls_count = 0
            min_theta = xg
            min_g2 = gg
            for _ in range(15):
                new_theta = xg - alpha * sign_gradient
                new_theta /= LA.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local_targeted(
                    model, x0, y0, yt, new_theta, initial_lbd = min_g2, tol=beta/500)
                ls_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= gg:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = xg - alpha * sign_gradient
                    new_theta /= LA.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local_targeted(
                        model, x0, y0, yt, new_theta, initial_lbd = min_g2, tol=beta/500)
                    ls_count += count
                    if new_g2 < gg:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        break
                        
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving")
                beta = beta*0.1
                if (beta < 1e-8):
                    break
            
            xg, gg = min_theta, min_g2
            
            query_count += (grad_queries + ls_count)
            #=================================
            D[nq:query_count] = gg
            nq = query_count
            #=================================
            ls_total += ls_count
            distortions.append(gg)

            if query_count > query_limit:
                print('break due to query cnt > query limit',query_count)
                break
            
            if i%10==0:
                print("Iteration %3d distortion %.4f num_queries %d" % (i+1, gg, query_count))
#                 print("Iteration: ", i, " Distortion: ", gg, " Queries: ", query_count,
#                       " LR: ", alpha, "grad_queries", grad_queries, "ls_queries", ls_count)
            if auto_terminate:
                if i%self.len_T==0:
                    DR = gg
                    if (DL-DR) < self.delta:
                        print('\n break due to slow convergence and group dim = 1!\n')
                        break
                    else:
                        DL = DR
            #if distortion is not None and gg < distortion:
            #    print("Success: required distortion reached")
            #    break

#             if gg > prev_obj-stopping:
#                 print("Success: stopping threshold reached")
#                 break 
#             prev_obj = gg

        adv_target = model.predict_label(x0 + torch.tensor(gg*xg, dtype=torch.float).cuda())
        if (adv_target == yt):
            timeend = time.time()
            print("\nAdversarial Example Found Successfully: distortion %.4f target"
                  " %d queries %d LS queries %d \nTime: %.4f seconds" % (gg, adv_target, query_count, ls_total, timeend-timestart))

            return x0 + torch.tensor(gg*xg, dtype=torch.float).cuda(), query_count,D[:query_count]
        else:
            print("Failed to find targeted adversarial example.")
            return x0 
    
    def fine_grained_binary_search_local_targeted(self, model, x0, y0, t, theta, initial_lbd=1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd

        if model.predict_label(x0 + torch.tensor(lbd*theta, dtype=torch.float).cuda()) != t:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while model.predict_label(x0 + torch.tensor(lbd_hi*theta, dtype=torch.float).cuda()) != t:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 100: 
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while model.predict_label(x0 + torch.tensor(lbd_lo*theta, dtype=torch.float).cuda()) == t:
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).cuda()) == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid

#         temp_theta = np.abs(lbd_hi*theta)
#         temp_theta = np.clip(temp_theta - 0.15, 0.0, None)
#         loss = np.sum(np.square(temp_theta))
        return lbd_hi, nquery

    def fine_grained_binary_search_targeted(self, model, x0, y0, t, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if model.predict_label(x0 + torch.tensor(current_best*theta, dtype=torch.float).cuda()) != t:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd
        
        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).cuda()) != t:
                lbd_lo = lbd_mid
            else:
                lbd_hi = lbd_mid
        return lbd_hi, nquery 
    
        