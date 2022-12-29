# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 18:24:24 2022

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

# 設定超參數
WAV_DIR = 'C:\\Users\\Gavin\\OneDrive\\桌面\\深度學習智慧應用\Lab 5\Lab5\Dataset\\LibriSpeech\\train-clean-100\\'
DATASET_DIR = 'C:\\Users\\Gavin\\OneDrive\\桌面\\深度學習智慧應用\Lab 5\Lab5\Dataset\\LibriSpeech\\train-clean-100-npy\\'

BATCH_SIZE = 32       
TRIPLET_PER_BATCH = 3

NUM_FRAMES = 160   
SAMPLE_RATE = 16000
TRUNCATE_SOUND_SECONDS = (0.2, 1.81)  # (start_sec, end_sec)
NUM_SPEAKERS = 251
EMBEDDING_SIZE = 512

from glob import glob
import os
from pydub import AudioSegment
from tqdm import tqdm

# 檢查是否解壓縮成功
print(os.path.exists("LibriSpeech/train-clean-100/19/198/19-198-0000.flac"))

if not os.path.exists("train-clean-100-npy"):
    files = glob(os.path.join(WAV_DIR, '**\\**\\*.flac'), recursive=True)

    for file in tqdm(files):
        print(file)
        tmp = AudioSegment.from_file(file, format='flac')
        tmp.export(file.replace('.flac', '.wav'), format='wav')