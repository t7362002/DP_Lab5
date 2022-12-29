# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 22:11:28 2022

@author: Gavin
"""

'''
注意資料結構如下：

LibriSpeech
  |----train-clean-100
    |----19
      |----198
        |----19-198-trans.txt
        |----19-198-0001.flac
        |----19-198-0002.flac
  |----train-clean-100-npy
'''
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'''
#from tabulate import tabulate
from scipy import stats
import heapq
import time
from pdb import set_trace as bp

import shutil
import os
import math
import logging
import librosa
import numpy as np
import pandas as pd
from glob import glob
from python_speech_features import fbank, delta
from tqdm import tqdm
from time import time
import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.transforms as transforms
import random
import sys
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from pydub import AudioSegment
from os.path import exists

# 設定超參數
WAV_DIR = './LibriSpeech/train-clean-100/'
DATASET_DIR = './LibriSpeech/train-clean-100-npy/'

BATCH_SIZE = 64     
TRIPLET_PER_BATCH = 3

NUM_FRAMES = 160   
SAMPLE_RATE = 16000
TRUNCATE_SOUND_SECONDS = (0.2, 1.81)  # (start_sec, end_sec)
NUM_SPEAKERS = 251
EMBEDDING_SIZE = 512

m_size = 0.2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_index = np.zeros(NUM_SPEAKERS, dtype=int)
if exists("./train_index.npy"):
    train_index = np.load("./train_index.npy")
    

#Feature Extraction
class SilenceDetector(object):
    def __init__(self, threshold=20, bits_per_sample=16):
        self.cur_SPL = 0
        self.threshold = threshold
        self.bits_per_sample = bits_per_sample
        self.normal = pow(2.0, bits_per_sample - 1)
        self.logger = logging.getLogger('balloon_thrift')

    def is_silence(self, chunk):
        self.cur_SPL = self.soundPressureLevel(chunk)
        is_sil = self.cur_SPL < self.threshold
        # print('cur spl=%f' % self.cur_SPL)
        if is_sil:
            self.logger.debug('cur spl=%f' % self.cur_SPL)
        return is_sil

    def soundPressureLevel(self, chunk):
        value = math.pow(self.localEnergy(chunk), 0.5)
        value = value / len(chunk) + 1e-12
        value = 20.0 * math.log(value, 10)
        return value

    def localEnergy(self, chunk):
        power = 0.0
        for i in range(len(chunk)):
            sample = chunk[i] * self.normal
            power += sample*sample
        return power
    
def find_files(directory, pattern='**/*.wav'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True) 
    
def VAD(audio):
    chunk_size = int(SAMPLE_RATE*0.05) # 50ms
    index = 0
    sil_detector = SilenceDetector(15)
    nonsil_audio=[]
    while index + chunk_size < len(audio):
        if not sil_detector.is_silence(audio[index: index+chunk_size]):
            nonsil_audio.extend(audio[index: index + chunk_size])
        index += chunk_size

    return np.array(nonsil_audio)

def read_audio(filename, sample_rate=SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    # print(audio.shape[0])
    audio = VAD(audio.flatten())
    # print(audio.shape[0])
    start_sec, end_sec = TRUNCATE_SOUND_SECONDS
    start_frame = int(start_sec * SAMPLE_RATE)
    end_frame = int(end_sec * SAMPLE_RATE)

    if len(audio) < (end_frame - start_frame):
        au = [0] * (end_frame - start_frame)
        for i in range(len(audio)):
            au[i] = audio[i]
        audio = np.array(au)
    # print(audio.shape[0])
    return audio

def normalize_frames(m,epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v),epsilon) for v in m]

def extract_features(signal=np.random.uniform(size=48000), target_sample_rate=SAMPLE_RATE):
    filter_banks, energies = fbank(signal, samplerate=target_sample_rate, nfilt=64, winlen=0.025)  #filter_bank (num_frames , 64),energies (num_frames ,)
    filter_banks = normalize_frames(filter_banks)
    frames_features = filter_banks     # (num_frames , 64)
    num_frames = len(frames_features)  # (num_frames)
    return np.reshape(np.array(frames_features),(num_frames, 64, 1))   #(num_frames,64, 1)

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

def prep(libri,out_dir=DATASET_DIR):
    if not os.path.exists(out_dir):
      os.mkdir(out_dir)
    # i=0
    for i in tqdm(range(len(libri))):
        filename = libri[i:i+1]['filename'].values[0] # for example: LibriSpeech/train-clean-100/1/100/1-100-0001.wav
        target_filename = out_dir + filename.split("/")[-1].split('.')[0] + '.npy' # for example: LibriSpeech/train-clean-100-npy/1-100-0001.npy
        # 確認是否已經產生.npy檔了
        if os.path.exists(target_filename):
          continue
        
        fp = open(target_filename,'w')  
        fp.close()
        raw_audio = read_audio(filename)
        feature = extract_features(raw_audio, target_sample_rate=SAMPLE_RATE)
        if feature.ndim != 3 or feature.shape[0] < NUM_FRAMES or feature.shape[1] !=64 or feature.shape[2] != 1:
            print('there is an error in file:',filename)
            continue
        np.save(target_filename, feature)

def preprocess_and_save(wav_dir=WAV_DIR,out_dir=DATASET_DIR):

    orig_time = time()
    libri = data_catalog(wav_dir, pattern='**/*.wav') 

    print("Extract fbank from audio and save as npy")
    prep(libri,out_dir)
    print("Extract audio features and save it as npy file, cost {0} seconds".format(time()-orig_time))

#Model & Loss Function
class PairwiseDistance(Function):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        # assert x1.size() == x2.size()
        #x1 = torch.unsqueeze(x1, 1) # 30 1 512
        #x2 = torch.unsqueeze(x2, 1)
        #diff = torch.bmm(x1.view(x1.shape[0],1,512),x2.view(x2.shape[0],512,1))
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)     
        # return diff
    
class TripletMarginLoss(Function):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2:歐式距離(if use 1:曼哈頓距離)

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)
        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.sum(dist_hinge)
        return loss


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

#Homework -- stochastic_mini_batch
"""
   filename                             chapter_id   speaker_id
0  1272/128104/1272-128104-0000.wav     128104       1272
1  1272/128104/1272-128104-0001.wav     128104       1272
2  1272/128104/1272-128104-0002.wav     128104       1272
3  1272/128104/1272-128104-0003.wav     128104       1272
4  1272/128104/1272-128104-0004.wav     128104       1272
5  1272/128104/1272-128104-0005.wav     128104       1272
6  1272/128104/1272-128104-0006.wav     128104       1272
7  1272/128104/1272-128104-0007.wav     128104       1272
8  1272/128104/1272-128104-0008.wav     128104       1272
9  1272/128104/1272-128104-0009.wav     128104       1272
"""
class stochastic_mini_batch(Dataset):
    #  [Dataset]
    #  index   filename                                            speaker_id   chapter_id  chapter_child_id
    #   0    LibriSpeech/train-clean-100-npy/1-100-0001.npy        1            100         0001 
    #   1    LibriSpeech/train-clean-100-npy/1-100-0002.npy        1            100         0002  
    
    def __init__(self,  libri, speaker_l):
       
        self.libri=libri
        self.speaker_list = speaker_l
        self.speaker_index = -1
        #self.have_train = np.zeros(251,251)
        #for i in range(0,251):
        #    self.have_train[i][i] = np.inf

    def clipped_audio(self, x, num_frames=NUM_FRAMES):
        if x.shape[0] > num_frames:
            bias = np.random.randint(0, x.shape[0] - num_frames)
            clipped_x = x[bias: num_frames + bias]
        else:
            clipped_x = x

        return clipped_x
        
    def __len__(self):
        return len(self.libri)
    
    def __getitem__(self, index):
      
        ################ you should write here ################
        # hint 
        # 1. sample anchor file(==speaker_id,==chapter_id) ,positive file(==speaker_id,!=chapter_id), negative file(!=speaker_id)
        anchor = self.libri['speaker_id'][index]
        chapter = self.libri['chapter_id'][index]
        
        anchor_file = np.load(self.libri['filename'][index])
        #print(anchor_file)
        #anchor_file = torch.from_numpy(anchor_file)
        anchor_label = anchor
        
        #libri_c = libri.copy()
        libri_c = libri
        
        fliter_anchor = libri_c['speaker_id'] == anchor
        anchor_df = libri_c[fliter_anchor]
        
        fliter_positive = anchor_df['chapter_id'] != chapter
        positive_df = anchor_df[fliter_positive]  #positive_df OK!
        positive_df.reset_index(drop=True, inplace=True)
        
        fliter_chapter = anchor_df['chapter_id'] == chapter
        anchor_df = anchor_df[fliter_chapter] #anchor_df OK!
        anchor_df.reset_index(drop=True, inplace=True)

        #fliter_negative = libri_c['speaker_id'] != anchor
        index = np.where(self.speaker_list==anchor)[0][0]
        self.speaker_index = train_index[index]
        if self.speaker_list[self.speaker_index] == anchor:
            self.speaker_index = (self.speaker_index + 1) % 251
        fliter_negative = libri_c['speaker_id'] == self.speaker_list[self.speaker_index]
        self.speaker_index = (self.speaker_index + 1) % 251
        train_index[index] = self.speaker_index
        
        #print(train_index)
        
        negative_df = libri_c[fliter_negative] #negative_df OK!
        negative_df.reset_index(drop=True, inplace=True)

        #positive_df = self.clipped_audio(positive_df,anchor_df.shape[0])
        #negative_df = self.clipped_audio(negative_df,anchor_df.shape[0])
        anchor_df.reset_index(drop=True, inplace=True)
        positive_df.reset_index(drop=True, inplace=True)
        negative_df.reset_index(drop=True, inplace=True)
        #print(positive_df)
        
        #print("positive_df.shape[0]", positive_df.shape[0])
        #print("index:", index, anchor)
        #print("a_ch:", chapter)
        if positive_df.shape[0]==0:
            pred_index = np.random.randint(0,anchor_df.shape[0])
            positive_file = np.load(anchor_df['filename'][pred_index])
            #positive_file = torch.from_numpy(positive_file)
            positive_label = anchor_df['speaker_id'][pred_index]
        else:
            pred_index = np.random.randint(0,positive_df.shape[0])
            positive_file = np.load(positive_df['filename'][pred_index])
            #positive_file = torch.from_numpy(positive_file)
            positive_label = positive_df['speaker_id'][pred_index]
            #print("p_ch:", positive_df['chapter_id'][0], positive_df['filename'][0])
        
        pred_index = np.random.randint(0,negative_df.shape[0])
        negative_file = np.load(negative_df['filename'][pred_index])
        #negative_file = torch.from_numpy(negative_file)
        negative_label = negative_df['speaker_id'][pred_index]
        
        #min_size = min(anchor_file.shape[0],positive_file.shape[0],negative_file.shape[0])
        min_size = 160
        anchor_file = self.clipped_audio(anchor_file,min_size)
        positive_file = self.clipped_audio(positive_file,min_size)
        negative_file = self.clipped_audio(negative_file,min_size)
        #print(anchor_file.shape[0],positive_file.shape[0],negative_file.shape[0])
        anchor_file = torch.from_numpy(anchor_file.transpose((2,0,1)))
        positive_file = torch.from_numpy(positive_file.transpose((2,0,1)))
        negative_file = torch.from_numpy(negative_file.transpose((2,0,1)))
        #print(anchor_file)
        '''
        libri_c = libri.copy()
        cid_list = []
        for index,row in libri_c.iterrows():
            cid_list.append(row['filename'].split('/')[-1].split('-')[1])
        libri_c['chapter_id'] = cid_list
        
        anchor = libri_c['speaker_id'][index]
        chapter = libri_c['chapter_id'][index]
        
        fliter_anchor = libri_c['speaker_id'] == anchor
        anchor_df = libri_c[fliter_anchor]
        
        fliter_positive = libri_c['chapter_id'] != chapter
        positive_df = libri_c[fliter_positive]  #positive_fd OK!
        
        fliter_chapter = libri_c['chapter_id'] == chapter
        anchor_df = libri_c[fliter_chapter] #anchor_df OK!

        fliter_negative = libri_c['speaker_id'] != anchor
        negative_df = libri_c[fliter_negative] #negative_df OK!

        #positive_df = clipped_audio(positive_df,anchor_df.shape[0])
        #negative_df = clipped_audio(negative_df,anchor_df.shape[0])

        anchor_file = []
        positive_file = []
        negative_file = []
        '''
        # 2. np.load(...)
        # 3. clipped_audio(...)
        # 4. torch.from_numpy(....transpose ((2, 0, 1)))
        #print(index, anchor_file, positive_file, negative_file, anchor_label, positive_label, negative_label)
        return anchor_file, positive_file, negative_file, anchor_label, positive_label, negative_label
    
#Train
def create_dict(files,labels,spk_uniq):
    train_dict = {}
    for i in range(len(spk_uniq)):
        train_dict[spk_uniq[i]] = []
    for i in range(len(labels)):
        train_dict[labels[i]].append(files[i])
    for spk in spk_uniq:
        if len(train_dict[spk]) < 2:
            train_dict.pop(spk)

    unique_speakers=list(train_dict.keys())
    return train_dict, unique_speakers

def load_model(model_path):
    model = DeepSpeakerModel(embedding_size=EMBEDDING_SIZE, num_classes=NUM_SPEAKERS)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    
    print('=> loading checkpoint')
    checkpoint = torch.load(model_path)    
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.cuda()
    return model, optimizer

def train(model, train_loader, optimizer):
  epoch = 0
  model.cuda()
  summary(model, input_size=(1, 160, 64))
  for epoch in range(100):       
    model.train()
    for batch_idx, (data_a, data_p, data_n, label_a, label_p, label_n) in tqdm(enumerate(train_loader)):
      
      data_a, data_p, data_n = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor), data_n.type(torch.FloatTensor)
      data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
      data_a, data_p, data_n = Variable(data_a), Variable(data_p), Variable(data_n)
      #print(data_a)
      out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)
      
      triplet_loss = TripletMarginLoss(m_size).forward(out_a, out_p, out_n)
      loss = triplet_loss
      # compute gradient and update weights
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print('selected_triplet_loss', triplet_loss.data)
    print("epoch:",epoch)
    # torch.save(model.state_dict(),"checkpoint_{}.pt".format(epoch))
    torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, "checkpoint_{}.pt".format(epoch))

#Evaluate
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

#flac轉wav
'''
# 檢查是否解壓縮成功
print(os.path.exists("LibriSpeech/train-clean-100/19/198/19-198-0000.wav"))

if not os.path.exists("train-clean-100-npy"):
  files = glob(os.path.join(WAV_DIR, '**/**/*.flac'), recursive=True)

  for file in tqdm(files):
    tmp = AudioSegment.from_file(file, format='flac')
    tmp.export(file.replace('.flac', '.wav'), format='wav')
    
print(os.path.exists("LibriSpeech/train-clean-100/19/198/19-198-0000.wav"))
'''

# extract fbanck from wav and save to file
# pre processd an audio in 0.09912s
np.set_printoptions(threshold=10)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)

'''
資料結構如下：
LibriSpeech
  |----train-clean-100
    |----19
      |----198
        |----19-198-trans.txt
        |----19-198-0001.flac
        |----19-198-0001.wav
        |----19-198-0002.flac
        |----19-198-0002.wav
  |----train-clean-100-npy
    |----19-198-0001.npy
    |----19-198-0002.npy
'''
#preprocess_and_save(WAV_DIR, DATASET_DIR)   #.wav --> .npy

#Train

print('Looking for fbank features [.npy] files in {}.'.format(DATASET_DIR))
libri = data_catalog(DATASET_DIR)
#  index   filename                                            speaker_id   chapter_id  chapter_child_id
#   0    LibriSpeech/train-clean-100-npy/1-100-0001.npy        1            100         0001 
#   1    LibriSpeech/train-clean-100-npy/1-100-0002.npy        1            100         0002
#print(libri)
unique_speakers = libri['speaker_id'].unique() # 251 speaker


#print("Training Speaker", unique_speakers[:4])
#mask1 = libri['speaker_id'].isin(unique_speakers[:3])
#mask2 = libri['speaker_id'] == 26
#libri = libri[mask1]
#print(libri)                            
train_dir = stochastic_mini_batch(libri, unique_speakers)
train_loader = DataLoader(train_dir, batch_size=BATCH_SIZE, shuffle=True)

# Retraining
ckpt_path = ""#"./checkpoint_99.pt"
if ckpt_path != "":
  model, optimizer = load_model(ckpt_path)
else:
  model = DeepSpeakerModel(embedding_size=EMBEDDING_SIZE, num_classes=NUM_SPEAKERS)
  optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

train(model, train_loader, optimizer)
np.save("./train_index.npy", train_index)


#Evaluate
'''
model = DeepSpeakerModel(embedding_size=512,num_classes=251)
# Load your model
checkpoint = torch.load("checkpoint_99.pt",map_location='cpu')
model.load_state_dict(checkpoint, strict=False)

#讀取資料庫
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
    print(labels)
    print(temp_embedding)
    np.save('emb',temp_embedding)
    np.save('emb_label',labels)
    
database = Database(20000)

for i in range(len(labels)):
    test_array, test_label = temp_embedding[i],labels[i] 
    database.insert(test_label, test_array)
print("inserting database completed")

#Homework -- 以 "inference-npy" inference 資料庫並計算準確率
#Accuracy = 預測正確數 / 資料總數
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
    print(temp_embedding2)
    np.save('emb2',temp_embedding2)
    np.save('emb2_label',labels2)
######## you should write here########
# nearest neighbor  mapping
# compute speaker identification accuracy
dis_list = []
all_pred_amount = 0
all_right_amount = 0
for i in range(temp_embedding2.shape[0]):
    #pred_amount = 0
    #right_amount = 0
    sim_list = []
    for j in range(temp_embedding.shape[0]):
        x = temp_embedding2[i]
        y = temp_embedding[j]
        #x = torch.from_numpy(temp_embedding2[i])
        #y = torch.from_numpy(temp_embedding[j])
        #print(x.shape,'\n',x)
        #print(y.shape,'\n',y)
        #dist_matrix = torch.pairwise_distance(x, y)
        #print(dist_matrix)
        #dis = dist_matrix.cpu().detach().numpy()
        #print(dis)
        sim_ans = get_similarity(x,y)
        sim_list.append(sim_ans)
    #min_dis = min(dis_list)
    max_sim = max(sim_list)
    #pred_anchor = labels[dis_list.index(min_dis)]
    pred_anchor = labels[sim_list.index(max_sim)]
    ans_anchor = labels2[i]
    print("label:",ans_anchor,"predict:",pred_anchor)
    all_pred_amount += 1
    if pred_anchor == ans_anchor:
        all_right_amount +=1
print(float(all_right_amount)/float(all_pred_amount))
'''









'''
#加入chapter_id
libri_c = libri.copy()
cid_list = []
ccid_list = []
for index,row in libri_c.iterrows():
    cid_list.append(row['filename'].split('/')[-1].split('-')[1])
    ccid_list.append(row['filename'].split('/')[-1].split('-')[2])
libri_c['chapter_id'] = cid_list
libri_c['chapter_child_id'] = ccid_list
'''