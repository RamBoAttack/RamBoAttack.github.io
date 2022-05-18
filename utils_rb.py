'''
- rev 0.1.0:
- rev 0.2.0:  
- rev 0.3.0: change PretrainedModel class to work with pretrained model on 
             unnorm dataset and path to user defined path for pretrained models
- rev 0.4.0: add new functions to export data using pandas

'''
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.utils.data as data
import numpy as np
import random
import pandas as pd
from xmodels import *
import torch.backends.cudnn as cudnn

# ======================== Dataset ========================

def load_data(dataset, data_path=None, batch_size=1):

    if dataset == 'imagenet' and data_path==None:
        data_path = '../datasets/ImageNet-val'
    elif dataset == 'cifar10'and data_path==None:
        data_path = '../datasets/cifar10' 

    if dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    elif dataset == 'cifar100':
        transform = transforms.Compose([transforms.ToTensor()])
        testset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    elif dataset == 'imagenet':
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])

        #testset = datasets.ImageNet(root=data_path, split='val', download=False, transform=transform)
        testset = datasets.ImageNet(root=data_path, split='val',  transform=transform)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return testloader, testset


# ======================== Model ========================

class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.features = self._make_layers()
        self.fc1 = nn.Linear(3200,256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256,256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256,10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def _make_layers(self):
        layers=[]
        in_channels= 3
        layers += [nn.Conv2d(in_channels, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.Conv2d(64, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(64, 128, kernel_size=3),
                   nn.BatchNorm2d(128),
                   nn.ReLU()]
        layers += [nn.Conv2d(128, 128, kernel_size=3),
                   nn.BatchNorm2d(128),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        return nn.Sequential(*layers)


    def predict(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True).view(1,3, 32,32)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]
    

class PretrainedModel():
    def __init__(self,model,dataset='imagenet',unnorm=False):
        self.model = model
        self.dataset = dataset
        self.unnorm = unnorm
        
        # ======= non-normalized =========       
        if self.unnorm:
            self.mu = torch.Tensor([0., 0., 0.]).float().view(1, 3, 1, 1).cuda()
            self.sigma = torch.Tensor([1., 1., 1.]).float().view(1, 3, 1, 1).cuda()
        
        else:
            # ======= CIFAR10 ==========
            if self.dataset == 'cifar10':
                self.mu = torch.Tensor([0.4914, 0.4822, 0.4465]).float().view(1, 3, 1, 1).cuda()
                self.sigma = torch.Tensor([0.2023, 0.1994, 0.2010]).float().view(1, 3, 1, 1).cuda()
            
            # ======= CIFAR100 =========
            elif self.dataset == 'cifar100':
                self.mu = torch.Tensor([0.5071, 0.4865, 0.4409]).float().view(1, 3, 1, 1).cuda()
                self.sigma = torch.Tensor([0.2673, 0.2564, 0.2762]).float().view(1, 3, 1, 1).cuda()
            
            # ======= ImageNet =========
            elif self.dataset == 'imagenet':
                self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1).cuda()
                self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1).cuda()              
            
            # ======= MNIST =========
            elif self.dataset == 'mnist':
                self.mu = torch.Tensor([0., 0., 0.]).float().view(1, 1, 1, 1).cuda()
                self.sigma = torch.Tensor([1., 1., 1.]).float().view(1, 1, 1, 1).cuda()          

    def predict(self, x):
        
        if len(x.size())!=4:
            x = x.unsqueeze(0)
        
        img = (x - self.mu) / self.sigma
        with torch.no_grad():
            out = self.model(img)
        return  out

    def predict_label(self, x):
        
        if len(x.size())!=4:
            x = x.unsqueeze(0)

        img = (x - self.mu) / self.sigma
        with torch.no_grad():
            out = self.model(img)
        out = torch.max(out,1)
        return out[1]

    def __call__(self, x):
        return self.predict(x)

def load_model(net,model_path=None):

    if net == 'resnet50':
        net = models.resnet50(pretrained=True).cuda()
    elif net == 'cifar10' :
        if model_path == None:
            model_path = './models/cifar10_gpu.pt'
        net = CIFAR10()
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        net = net.to(device)
        net = torch.nn.DataParallel(net, device_ids=[0])

        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint)
        net = net.module # due to the model is save in a special way. So need to select submodule from the model "net"
    '''
    else:
        if model_path == None:
            model_path = './models/cifar10_ResNet18.pth'
        net = ResNet18()
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        net = net.to(device)
        checkpoint = torch.load(model_path,map_location='cuda:0')
        if 'net' in checkpoint:
            if device == 'cuda:0':
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True
            net.load_state_dict(checkpoint['net'])
        else:
            net.load_state_dict(checkpoint)
    '''
    net.eval()
    return net


class PytorchModel_ex(object):
    def __init__(self,model, bounds, num_classes,dataset,unnorm=False):
        self.model = model
        self.model.eval()
        self.bounds = bounds
        self.num_classes = num_classes
        self.num_queries = 0
        self.dataset = dataset
        self.unnorm = unnorm
    
    def predict_label(self, image, batch=False):

        # convert "numpy" to "torch"
        #if isinstance(image, np.ndarray):
        #    image = torch.from_numpy(image).type(torch.FloatTensor)
        
        # clamp and send to CUDA
        #image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
        image = torch.clamp(image,self.bounds[0],self.bounds[1])
        
        # convert from 3 to 4
        if len(image.size())!=4:
            image = image.unsqueeze(0)
        #image = Variable(image, volatile=True) # ?? not supported by latest pytorch

        # normalize image before process classification
        # ======================================================
        n = len(image)
        #print('Shape in pytorch_model batch prediction:',image.shape, n)
        #norm_img = torch.zeros(image.shape).cuda()
        if self.unnorm:
            mean = (0.0, 0.0, 0.0)
            std = (1.0, 1.0, 1.0)

        else:
            if self.dataset == 'cifar10':
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)

            elif self.dataset == 'imagenet':
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225) 

        normalize = transforms.Normalize(mean,std)

        for i in range(n):
            #norm_img[i] = normalize(image[i])
            image[i] = normalize(image[i])

        # ======================================================
        
        with torch.no_grad():
            output = self.model(image)
            self.num_queries += image.size(0)
             
        #image = Variable(image, volatile=True) # ?? not supported by latest pytorch
        _, predict = torch.max(output.data, 1)
        if batch:
            return predict
        else:
            return predict[0] 

# ======================== Load pre-defined ImageNet or CIFAR-10 subset ========================

def load_predefined_set(filename,targeted):
    df = pd.read_csv(filename, index_col=0)
    if targeted:
        # out: [ocla, oID, tcla, tID]
        np_df = df.to_numpy()
    else:
        # out: [ocla, oID]
        np_df = df.drop(['tcla', 't_ID'], axis=1)
        np_df = np_df.drop_duplicates(subset=['ocla','o_ID'],keep='first')
        np_df = np_df.to_numpy()

    return np_df

def get_evalset(dataset,targeted,eval_set):
    if dataset == 'imagenet':
        if eval_set == 'balance':
            subset_path = './evaluation_set/ImageNet - balance common set - final.csv'
        elif eval_set == 'hardset':
            subset_path = './evaluation_set/ImageNet - hardset - final.csv'
        elif eval_set == 'easyset':
            subset_path = './evaluation_set/ImageNet - easyset - final.csv'
                
    elif dataset == 'cifar10':
        if eval_set == 'balance':
            subset_path = './evaluation_set/cifar10 - balance common set - final.csv'
        elif eval_set == 'hardset_A':
            subset_path = './evaluation_set/cifar10 - hardset A (BA) - final.csv'
        elif eval_set == 'hardset_B':
            subset_path = './evaluation_set/cifar10 - hardset B (HSJA-SignOPT) - final.csv'
        elif eval_set == 'easyset':
            subset_path = './evaluation_set/cifar10 - easyset C - final.csv'
        elif eval_set == 'hardset_D':
            subset_path = './evaluation_set/cifar10 - hardset D (RamBo) - final.csv'

    output = load_predefined_set(subset_path,targeted)

    return output

# ========================= Export to csv ======================

def export_pd_csv(D,head,key_info,output_path,n_point=None,query_limit=None):

    if n_point is not None:
        #1.
        key = pd.DataFrame(key_info)
        step_size = int(query_limit/n_point)

        #2.
        data = np.zeros(n_point)#.astype(int)  
        for k in range(n_point):
            q = k*step_size
            if q<(len(D)):
                data[k]= D[q]
            else:
                data[k]= D[len(D)-1]

        #3.
        out = pd.concat([key.transpose(), pd.DataFrame(data).transpose()], axis=1).to_numpy()

    else:
        key = key_info.copy()
        key.append(str(D))
        out = np.array(key).reshape(1,-1)

    #4
    df = pd.DataFrame(out,columns = head)
    with open(output_path, mode = 'a') as f:
        df.to_csv(f, header=f.tell()==0,index = False)
