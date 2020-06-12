import math
import os
import sys
import numpy as np
import cv2
import torch
from torch import nn
import cifar10_models
import torch.optim as optim
from torchvision import models
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F
from misc_functions import preprocess_image, recreate_image, get_params
from DeepFool.Python.deepfool import deepfool
import matplotlib.pyplot as plt
from PIL import Image

device='cpu'

class cifar(torch.utils.data.Dataset):
    def __init__(self,data,label):
        self.images=data
        self.labels=torch.LongTensor(label)
    def __getitem__(self,index):
        img=self.images[index]
        img2=preprocess_image(img,source='cifar')
        label = self.labels[index]
        return img2, label

    def __len__(self):
        return (self.images.shape[0])

def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi
    if p==np.inf:
            v=torch.clamp(v,-xi,xi)
    else:
        v=v * min(1, xi/(torch.norm(v,p)+0.00001))
    return v

    
def get_fooling_rate(ima,lab,v,model, device):

    num_images = len(ima)
    trainset=cifar(ima,lab)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=False,num_workers=0)
    
    fooled=0.0
    
    for i,data in enumerate(trainloader,0):
        img,label=data
        img=img[0]
        _, adv_pred = torch.max(model(img+v),1)
        if(label!=adv_pred):
            fooled+=1

    # Compute the fooling rate
    fooling_rate = fooled/num_images
    print("Fooling Rate = ", fooling_rate)
    for param in model.parameters():
        param.requires_grad = False
    
    return fooling_rate,model


def universal_adversarial_perturbation(ima,lab, model, xi=10, delta=0.15, max_iter_uni = 10, p=np.inf, 
                                       num_classes=10, overshoot=0.02, max_iter_df=10,t_p = 0.2):

    v = torch.zeros(1,3,32,32).to(device)
    v.requires_grad_()
    #ima=ima[0:600]
    #lab=lab[0:600]
    fooling_rate = 0.0
    num_images =  len(ima)
    itr = 0
    trainset=cifar(ima,lab)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=False,num_workers=0)
    
    while fooling_rate < 1-delta and itr < max_iter_uni:
        # Iterate over the dataset and compute the purturbation incrementally
        print('Starting pass number ', itr)
        for i,data in enumerate(trainloader,0):
            img,label=data
            img=img[0]
            #_, pred = torch.max(model(img),1)
            _, adv_pred = torch.max(model(img+v),1)
            
            if(label==adv_pred):
                dr, iter, _,_,_ = deepfool((img+v).detach()[0],model, device, num_classes= num_classes,
                                             overshoot= overshoot,max_iter= max_iter_df)
                if(iter<max_iter_df-1):
                    v = v + torch.from_numpy(dr).to(device)
                    v = proj_lp(v,xi,p)
                    
            if(i%40==0):
                print(i,'/',num_images)
        fooling_rate,model = get_fooling_rate(ima,lab,v,model, device)
        itr = itr + 1
    
    return v

def showus(ima,lab):
    #ima=ima[0:600]
    #lab=lab[0:600]
    trainset=cifar(ima,lab)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=False,num_workers=0)
    for i,data in enumerate(trainloader,0):
        if((i+1)%98!=0):
            continue
        img,label=data
        img=img[0]
        _, pred = torch.max(model(img),1)
        __, adv_pred = torch.max(model(img+noise),1)

        plt.figure(dpi=256)
        plt.subplot(131)
        plt.axis('off')
        plt.imshow(img[0].detach().permute(1,2,0).cpu().numpy())
        plt.title('Original Image '+str(pred.cpu().numpy()))
        plt.subplot(132)
        plt.axis('off')
        plt.imshow(noise[0].detach().permute(1,2,0).cpu().numpy())
        plt.title('noise')
        plt.subplot(133)
        plt.axis('off')
        plt.imshow((img+noise)[0].detach().permute(1,2,0).cpu().numpy())
        plt.title('Adv Image '+str(adv_pred.cpu().numpy()))
        plt.show()
        plt.savefig("qaz.png")
        return

def check(ima,lab,v,model):
    trainset=cifar(ima,lab)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=False,num_workers=0)
    num_images = len(ima)
    fooled=0
    
    for i,data in enumerate(trainloader,0):
        img,label=data
        img=img[0]
        if(i%500==0):
            print("Check image:",i*100/num_images,"%")
        _, adv_pred = torch.max(model(img+v),1)
        if(label!=adv_pred):
            fooled+=1
    print("This batch Fooling Rate = ",float(fooled)/num_images)
    # Compute the fooling rate
    return (fooled,num_images)

if __name__ == '__main__':
    model = cifar10_models.vgg16_bn(pretrained=True)
    model.eval()
    model.to(device)

    noise=torch.load('noise.pth')
    #A=noise[0].detach().permute(1,2,0).cpu().numpy()
    #A=A.astype(np.uint8)
    #im = Image.fromarray(A)
    #im.save("noise.jpeg")
    
    fooled=0.0
    tot=0.0
    for index in range(2,3):
        ima=np.load(sys.path[0]+"/../data/databatch"+str(index)+".npy")
        lab=np.load(sys.path[0]+"/../data/label"+str(index)+".npy")
        
        showus(ima,lab)   
    
        #noise=universal_adversarial_perturbation(ima,lab,model)
        #print("Load databatch:",index)
        #x,y=check(ima,lab,noise,model)
        #fooled+=x
        #tot+=y

        #plt.figure(dpi=200)
        #plt.imshow(noise[0].detach().permute(1,2,0).cpu().numpy())
        #plt.title('qwq')
        #plt.axis('off')
        #plt.show()

        #torch.save(noise,'noise.pth')
        

    print("All Fooling Rate = ",fooled/tot,"tot:",tot)

