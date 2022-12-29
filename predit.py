# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 16:40:31 2022

@author: Gavin
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn

import os
import pandas as pd
from glob import glob
from torch.autograd import Variable

import numpy as np

DATASET_DIR = './LibriSpeech/train-clean-100-npy/'

def find_files(directory, pattern='**/*.wav'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True) 

def data_catalog(dataset_dir=DATASET_DIR, pattern='*.npy'): 
    libri = pd.DataFrame()                                            
    libri['filename'] = find_files(dataset_dir, pattern=pattern)
    libri['filename'] = libri['filename'].apply(lambda x: x.replace('\\', '/'))  # normalize windows paths
    libri['speaker_id'] = libri['filename'].apply(lambda x: x.split('/')[-1].split('-')[0]) # x.split('/')[-1]->1-100-0001.wav 
    libri['chapter_id'] = libri['filename'].apply(lambda x: x.split('/')[-1].split('-')[1]) # x.split('/')[-1]->1-100-0001.wav 
    libri['chapter_child_id'] = libri['filename'].apply(lambda x: x.split('/')[-1].split('-')[2]) # x.split('/')[-1]->1-100-0001.wav 
    num_speakers = len(libri['speaker_id'].unique())
    print('Found {} files with {} different speakers.'.format(str(len(libri)).zfill(7), str(num_speakers).zfill(5)))
    libri.to_csv('libri.csv')
    return libri
    #                 filename                                       speaker_id
    #   0    LibriSpeech/train-clean-100/1/100/1-100-0001.wav        1
    #   1    LibriSpeech/train-clean-100/1/100/1-100-0002.wav        1

class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class myResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):

        super(myResNet, self).__init__()

        self.relu = ReLU(inplace=True)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.inplanes = 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.inplanes = 256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.inplanes = 512
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.layer4 = self._make_layer(block, 512, layers[3])

        
        self.avgpool = nn.AdaptiveAvgPool2d((1,None))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

class DeepSpeakerModel(nn.Module):
    def __init__(self,embedding_size,num_classes,feature_dim = 64):
        super(DeepSpeakerModel, self).__init__()

        self.embedding_size = embedding_size
        self.model = myResNet(BasicBlock, [1, 1, 1, 1])
        if feature_dim == 64:
            self.model.fc = nn.Linear(512*4, self.embedding_size)
        elif feature_dim == 40:
            self.model.fc = nn.Linear(256 * 5, self.embedding_size)
        self.model.classifier = nn.Linear(self.embedding_size, num_classes)




    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)

        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer2(x)

        x = self.model.conv3(x)
        x = self.model.bn3(x)
        x = self.model.relu(x)
        x = self.model.layer3(x)

        x = self.model.conv4(x)
        x = self.model.bn4(x)
        x = self.model.relu(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.model.fc(x)
        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha=1
        self.features = self.features*alpha

        #x = x.resize(int(x.size(0) / 17),17 , 512)
        #self.features =torch.mean(x,dim=1)
        #x = self.model.classifier(self.features)
        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res

#from tabulate import tabulate
class Database():
    "Simulated data structure"
    def __init__(self, data_num):
        self.embs = np.ndarray((data_num,512), dtype=float)
        self.labels = []
        self.indices = 0
    

    def __len__(self):
        return self.indices

    def insert(self, label, emb,index=None):
        " Insert testing data "

        self.embs[self.indices] = emb
        self.labels.append(label)
        self.indices += 1

   
    def get_most_similar(self, embTest):
        testTiles = np.tile(embTest, (self.indices, 1))
        similarities = np.sum(testTiles*self.embs[0:self.indices], axis=1)
        max_similarity = np.max(similarities)
        max_id = np.argmax(similarities)
        return max_id, max_similarity,self.embs[max_id]
    

    def get_label_by_id(self, id):
        return self.labels[id]
    

def get_similarity(embA, embB): # inner product
    ans = np.sum(embA*embB)
    return ans

##-----Main-----##
acc = np.load("./acc.npy")
#print(acc.tostring())
#acc = np.zeros(100)

for j in range(66,67):
    model = DeepSpeakerModel(embedding_size=512,num_classes=251)
    # Load your model
    checkpoint = torch.load("checkpoint_"+str(j)+".pt",map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    
    libri = data_catalog("database-npy",pattern='*.npy')
    new_x = []
    labels = []
    model.eval()
    
    with torch.no_grad():
        flag=0
       
        for i in range(int(len(libri))):
            new_x = []
            filename =libri[i:i + 1]['filename'].values[0]
            filelabel=libri[i:i + 1]['speaker_id'].values[0]
            x = np.load(filename)
            if(x.shape[0]>160):
                for bias in range(0,x.shape[0]-160,160):
                    clipped_x = x[bias:bias+160]
                    new_x.append(clipped_x)
                    labels.append(filelabel)
            else:
                clipped_x = x
                new_x.append(clipped_x)
                labels.append(filelabel)
    
            x = np.array(new_x)
            print(x.shape)
            x_tensor = Variable(torch.from_numpy(x.transpose ((0,3, 1, 2))).type(torch.FloatTensor).contiguous())
            print(x_tensor.shape)
            embedding = model(x_tensor)
            if i == 0 :
                temp_embedding = embedding
            else :
                temp_embedding = torch.cat((temp_embedding,embedding),0)
    
        temp_embedding = temp_embedding.cpu().detach().numpy()
        labels=np.array(labels)
        labels = labels.astype("int32")
        print(labels.shape)
        print(temp_embedding.shape)
        np.save('emb',temp_embedding)
        np.save('emb_label',labels)
        
    database = Database(20000)
    
    for i in range(len(labels)):
        test_array, test_label = temp_embedding[i],labels[i] 
        database.insert(test_label, test_array)
    print("inserting database completed")
    
    
    libri = data_catalog("inference-npy",pattern='*.npy') #audio/LibriSpeechTest/test-clean-npy
    ########## you should write here ########
    # clipped
    # model and concat
    new_x = []
    labels2 = []
    model.eval()
    
    with torch.no_grad():
        flag=0
       
        for i in range(int(len(libri))):
            new_x = []
            filename =libri[i:i + 1]['filename'].values[0]
            filelabel=libri[i:i + 1]['speaker_id'].values[0]
            x = np.load(filename)
            if(x.shape[0]>160):
                for bias in range(0,x.shape[0]-160,160):
                    clipped_x = x[bias:bias+160]
                    new_x.append(clipped_x)
                    labels2.append(filelabel)
            else:
                clipped_x = x
                new_x.append(clipped_x)
                labels2.append(filelabel)
    
            x = np.array(new_x)
            print(x.shape)
            x_tensor = Variable(torch.from_numpy(x.transpose ((0,3, 1, 2))).type(torch.FloatTensor).contiguous())
            print(x_tensor.shape)
            embedding = model(x_tensor)
            if i == 0 :
                temp_embedding2 = embedding
            else :
                temp_embedding2 = torch.cat((temp_embedding2,embedding),0)
    
        temp_embedding2 = temp_embedding2.cpu().detach().numpy()
        labels2=np.array(labels2)
        labels2 = labels2.astype("int32")
        print(labels2.shape)
        print(temp_embedding2.shape)
        np.save('emb2',temp_embedding2)
        np.save('emb2_label',labels2)
        
    ######## you should write here########
    # nearest neighbor  mapping
    # compute speaker identification accuracy
    all_pred_amount = 0
    all_right_amount = 0
    for i in range(len(labels2)):
        all_pred_amount += 1
        infer_array, infer_label = temp_embedding2[i],labels2[i]
        max_id, max_similarity, max_emb = database.get_most_similar(infer_array)
        ans_label = database.get_label_by_id(max_id)
        print("label:",infer_label,"predict:",ans_label)
        if infer_label == ans_label:
            all_right_amount += 1
    print(float(all_right_amount)/float(all_pred_amount))
    acc[j] = float(all_right_amount)/float(all_pred_amount)
np.save("./acc.npy",acc)
print(acc)