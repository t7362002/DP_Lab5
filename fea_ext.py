# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 22:17:12 2022

@author: Gavin
"""

import math
import logging

# extract fbanck from wav and save to file
# pre processd an audio in 0.09912s

import os
import librosa
import numpy as np
import pandas as pd
from glob import glob
from python_speech_features import fbank, delta
from tqdm import tqdm
from time import time

# 設定超參數
WAV_DIR = './LibriSpeech/train-clean-100/'
DATASET_DIR = './LibriSpeech/train-clean-100-npy/'

BATCH_SIZE = 32       
TRIPLET_PER_BATCH = 3

NUM_FRAMES = 160   
SAMPLE_RATE = 16000
TRUNCATE_SOUND_SECONDS = (0.2, 1.81)  # (start_sec, end_sec)
NUM_SPEAKERS = 251
EMBEDDING_SIZE = 512

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.set_printoptions(threshold=10)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)

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
    num_speakers = len(libri['speaker_id'].unique())
    print('Found {} files with {} different speakers.'.format(str(len(libri)).zfill(7), str(num_speakers).zfill(5)))
    return libri
    #                          filename                                       speaker_id
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
preprocess_and_save(WAV_DIR, DATASET_DIR)