import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.models as models
from PIL import Image
import os
import numpy as np
from img2vec_pytorch import Img2Vec
from pytorch_pretrained_vit import ViT
from torchvision import models, transforms
import pretrainedmodels
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict

class ResNetImageEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        nn.Module.__init__(self)
        self.img2vec = Img2Vec(model='resnet-18', cuda=True)
    
        
    def get_image(self, img):
        filepath = os.getcwd() + '/data/image/' + img
        try:
            img = Image.open(filepath)
        except:
            print('error')
            img = Image.open(os.getcwd() + '/data/error.jpg')
        
        return img
    
    def forward(self, img):
        #print(img)
        data_array = []
        for b in range(len(img)):
            inputs = self.get_image(img[b])
            vec = self.img2vec.get_vec(inputs, tensor=True)
            vec = torch.squeeze(vec)
            data_array.append(vec)
        data_array = np.array(data_array)
        output = Variable(torch.from_numpy(np.stack(data_array, 0).astype(np.int64)).long())
        output = output.cuda()
        
        return output

    
    
class ViTImageEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        nn.Module.__init__(self)
        self.model = ViT('B_16_imagenet1k', pretrained=True)
        self.freeze_all_layers()
        self.unfreeze_fully_connected()
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.model.to(self.device)
        
    def freeze_all_layers(self):
        """Freeze the model so it's won't change the pretrained whights"""
        for key, module in self.model._modules.items():
            for param in module.parameters():
                param.requires_grad = False
                
    def unfreeze_fully_connected(self):
        """Unfreeze the whights of the last fully connected layer"""
        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True            
    
    def replace_fully_connected(self, block):
        """Replace the last fully connected layer with a given block"""
        self.model.fc = block
        
        
    
    
        
    def get_image(self, img):
        filepath = os.getcwd() + '/data/image/' + str(img).zfill(6) + '.jpg'
        try:
            img = Image.open(filepath)
        except:
            img = Image.open(os.getcwd() + '/data/error.jpg')
        
        return img
    
    def forward(self, img):
        data_array = []
        for b in range(img.size()[0]):
            for n in range(img.size()[1]):
                for k in range(img.size()[2]): 
                    image = self.get_image(img[b][n][k])
                    inputs = transforms.Compose([
    transforms.Resize((384, 384)), 
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])(image).unsqueeze(0).cuda()

                    
                    vec = self.model(inputs)
                    vec = torch.squeeze(vec)
                    data_array.append(vec)
        data_array = np.array(data_array)
        output = Variable(torch.from_numpy(np.stack(data_array, 0).astype(np.int64)).long())
        output = output.cuda()
        
        return output

    
    
    
class ResNetImageEncoder2(nn.Module):
    def __init__(self, embedding_dim=512):
        #super(EfficientNet_b0, self).__init__()
        nn.Module.__init__(self)
        #self.model = pretrainedmodels.__dict__['resnet18'](pretrained='imagenet')   
        self.resnet18 = models.resnet18(pretrained=True)
        self.newmodel = torch.nn.Sequential(*(list(self.resnet18.children())[:-1]))
        self.newmodel.eval()
        for param in self.newmodel.parameters():
            param.requires_grad = False
        self.num_ftrs = self.resnet18.fc.in_features
        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        
        self.classifier_layer = nn.Sequential(
            nn.Linear(self.num_ftrs, 512),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Linear(128, 128))
                                
    def get_image(self, img):
        filepath = os.getcwd() + '/data/image/' + img
        try:
            img = Image.open(filepath)
        except:
            print('error')
            img = Image.open(os.getcwd() + '/data/error.jpg')
        
        return img
    
    def get_vector(self, img):
        t_img = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0)).cuda()
        my_embedding = self.newmodel(t_img)
        my_embedding = torch.squeeze(my_embedding)
       
        #my_embedding = self.classifier_layer(my_embedding)
        return my_embedding
        
        
    def forward(self, img):
        data_array = []
        for b in range(len(img)):
            inputs = self.get_image(img[b])
            vec = self.get_vector(inputs)
            vec = torch.squeeze(vec).cpu().detach().numpy()
            data_array.append(vec)
        data_array = np.array(data_array)
        output = Variable(torch.from_numpy(np.stack(data_array, 0).astype(np.int64)).long())
        output = output.cuda()
        
        return output