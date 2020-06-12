import os
import sys
import numpy as np
import cv2
from torchvision import models
import torch
from torch import nn
from torch.autograd import Variable
# from torch.autograd.gradcheck import zero_gradients  # See processed_image.grad = None
from misc_functions import preprocess_image, recreate_image, get_params

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

    def generate(self, original_image, org_class, target_class, minimum, example_number):
        # I honestly dont know a better way to create a variable with specific value
        # Targeting the specific class
        im_label_as_var = Variable(torch.from_numpy(np.asarray([target_class])))
        # Define loss functions
        ce_loss = nn.CrossEntropyLoss()
        # Process image
        processed_image = preprocess_image(original_image)
        # Start iteration
        for i in range(30):
            print('launching epoch:', str(i))
            # zero_gradients(x)
            # Zero out previous gradients
            # Can also use zero_gradients(x)
            processed_image.grad = None
            # Forward pass
            out = self.model(processed_image)
            # Calculate CE loss
            pred_loss = ce_loss(out, im_label_as_var)
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
            recreated_image = recreate_image(processed_image)
            # Process confirmation image
            prep_confirmation_image = preprocess_image(recreated_image)
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
            #if confirmation_prediction == target_class:
            if confirmation_confidence > minimum:
                print('Original image was predicted as:', label[org_class],
                      '\nWith adversarial noise, converted to:', label[confirmation_prediction],
                      '\nclassification confidence: %.4f' % confirmation_confidence)
                # Create the image for noise as: Original image - generated image
                noise_image = original_image - recreated_image
                cv2.imwrite(sys.path[0]+'/../generated/'+ str(example_number)+'_noise_from_' + label[org_class] + '_to_' +
                            label[confirmation_prediction] + '.jpg', noise_image)
                # Write image
                cv2.imwrite(sys.path[0]+'/../generated/'+str(example_number)+'_adv_from_' + label[org_class] + '_to_' +
                            label[confirmation_prediction] + '.jpg', recreated_image)
                break

        return 1

if __name__ == '__main__':
    pretrained_model = models.vgg16(pretrained=True)
    target_example = 0 # cat
    (original_image, prep_img) = get_params(target_example)
    target_class=int(input("Input target class number: "))
    # minimum=float(input("Input minimal confidence of classification: "))
    minimum=0.97
    # target_class = 245  # bulldog
    label=np.load(sys.path[0]+'/label.npy').item()
    org_out=pretrained_model(prep_img)
    _,org_class=org_out.data.max(1)
    org_confidence=nn.functional.softmax(org_out,dim=1)[0][org_class].data.numpy()[0]
    org_class=org_class.numpy()[0]
    # print('Original picture:',label[org_class],' confidence: %.4f' % org_confidence)
    FGS_untargeted = FastGradientSignTargeted(pretrained_model, 0.01)
    FGS_untargeted.generate(original_image, org_class, target_class,minimum,target_example)
