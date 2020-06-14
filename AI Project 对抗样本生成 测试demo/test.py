import cv2
import glob

import torch
import torchvision.models as models
import torch.nn.functional as F

model = models.vgg16(pretrained=True)
model.eval()

def read_img(img_path):
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img = torch.from_numpy(img.transpose((2, 0, 1))).float()
	mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
	std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
	img = img.div_(255.0)
	img = (img - mean) / std
	return img.unsqueeze(0)

for img_path in glob.glob("*.png") + glob.glob("*.j*"):
	test_img = read_img(img_path)
	
	with torch.no_grad():
		feat = model(test_img)
	
	pred = F.softmax(feat)
	confidence, class_id = torch.max(pred, 1)

	print('IMG:%s, Conf:%.1f, Class:%d'%(img_path, confidence.item()*100, class_id.item()))


