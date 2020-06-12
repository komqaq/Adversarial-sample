import math
import os
import sys
import numpy as np
import cv2
import torch
from torch import nn
import torch.optim as optim
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import cifar10_models
# from torch.autograd.gradcheck import zero_gradients  # See processed_image.grad = None
from misc_functions import preprocess_image, recreate_image, get_params


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.eval()
        self.conv0=nn.Conv2d(3,16,kernel_size=1,stride=1)
        self.conv1 = nn.Conv2d(16,32, kernel_size=3, stride=1, padding=1)
        self.pool0=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(32,128,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1)
        self.pool1=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(4096,4096)
        self.fc2 = nn.Linear(4096, 3072)
    def forward(self,x):#x : processed image
        out=self.conv0(x)
        out=F.relu(self.conv1(out))
        out=self.pool0(out)
        out=F.relu(self.conv2(out))
        out=F.relu(self.conv3(out))
        out=self.pool1(out)
        out=self.fc1(out.view(out.size(0),-1))
        out=self.fc2(out)
        #out=torch.sign(out.view(-1,3,32,32))*0.03
        out=out.view(-1,3,32,32)
        return out

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
class FastGradientSignTargeted():
    """
        Fast gradient sign untargeted adversarial attack, maximizes the target class activation
        with iterative grad sign updates
    """
    def __init__(self, model, alpha):
        self.model = model
        self.model.eval()
        # Movement multiplier per epoch
        self.alpha = alpha
        # Create the folder to export images if not exists
        if not os.path.exists(sys.path[0]+'/../generated'):
            os.makedirs(sys.path[0]+'/../generated')

    def generate(self, processed_image, org_class,minimum=0.8):
        # I honestly dont know a better way to create a variable with specific value
        # Targeting the specific class
        im_label_as_var = Variable(torch.from_numpy(np.asarray([org_class])))
        # Define loss functions
        ce_loss = nn.CrossEntropyLoss()
        # Process image
        # Start iteration
        for i in range(30):
            ###print('launching epoch:', str(i))
            # zero_gradients(x)
            # Zero out previous gradients
            # Can also use zero_gradients(x)
            processed_image.grad = None
            # Forward pass
            out = self.model(processed_image)
            # Calculate CE loss
            pred_loss = -ce_loss(out, im_label_as_var)
            # Do backward pass
            pred_loss.backward()
            # Create Noise
            # Here, processed_image.grad.data is also the same thing is the backward gradient from
            # the first layer, can use that with hooks as well
            adv_noise = self.alpha * torch.sign(processed_image.grad.data)
            # Add noise to processed image
            processed_image.data = processed_image.data - adv_noise

            # Confirming if the image is indeed adversarial with added noise
            # This is necessary (for some cases) because when we recreate image
            # the values become integers between 1 and 255 and sometimes the adversariality
            # is lost in the recreation process

            # Generate confirmation image
            recreated_image = recreate_image(processed_image,source='cifar')
            # Process confirmation image
            prep_confirmation_image = preprocess_image(recreated_image,source='cifar')
            # Forward pass
            confirmation_out = self.model(prep_confirmation_image)
            # Get prediction
            _, confirmation_prediction = confirmation_out.data.max(1)
            # Get classification confidence of the result
            confirmation_confidence = \
                nn.functional.softmax(confirmation_out,dim=1)[0][confirmation_prediction].data.numpy()[0]
            # Convert tensor to int
            confirmation_prediction = confirmation_prediction.numpy()[0]
            # Check if the prediction is different than the original
            if (confirmation_prediction != org_class) and (confirmation_confidence>minimum):
                # Create the image for noise as: Original image - generated image
                break
        
        return adv_noise

if __name__ == '__main__':
    net=Net()
    net.load_state_dict(torch.load(sys.path[0]+"/otk.pth"))
    pretrained_model = cifar10_models.vgg16_bn(pretrained=True)
    cat2lab=np.load(sys.path[0]+'/label.npy').item()
    # print('Original picture:',label[org_class],' confidence: %.4f' % org_confidence)
    FGS_untargeted = FastGradientSignTargeted(pretrained_model, 0.05)
    #FGS_untargeted.generate(original_image, org_class,target_example)
    net.eval()
    legalcount=0
    for index in range(3,4):
        ima=np.load(sys.path[0]+"/../data/databatch"+str(index)+".npy")
        lab=np.load(sys.path[0]+"/../data/label"+str(index)+".npy")
        trainset=cifar(ima,lab)
        trainloader=torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=False,num_workers=0)
        count=0
        tot_noise_output=np.load(sys.path[0]+'/../data/batch1_noise.npy').astype(float)
        for i,data in enumerate(trainloader,0):
            count=count+1
            if (count>=500):break
            inputs,labels=data
            outputs=net(inputs)
            processed=inputs+outputs
            #temp=np.load(sys.path[0]+'/../qaz_train/'+str(i)+'_noise.npy')
            #processed=inputs+torch.Tensor(tot_noise_output[i])
            recreated_image = recreate_image(processed,source='cifar')
            prep_confirmation_image = preprocess_image(recreated_image,source='cifar').view(1,3,32,32)
            confirmation_out = pretrained_model(prep_confirmation_image)
            _, confirmation_prediction = confirmation_out.data.max(1)
            confirmation_confidence = nn.functional.softmax(confirmation_out,dim=1)[0][confirmation_prediction].data.numpy()[0]
            confirmation_prediction = confirmation_prediction.numpy()[0]
            if (confirmation_prediction != labels):
                legalcount=legalcount+1
            print(count," ",legalcount/count)
