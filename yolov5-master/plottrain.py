import math
import os
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from utils.general import (LOGGER, Timeout, check_requirements, clip_coords, increment_path, is_ascii, is_chinese,
                           try_except, user_config_dir, xywh2xyxy, xyxy2xywh)
from utils.metrics import fitness
def plot_results(file_5s='./runs/train/exp34/results.csv', file_5m='./runs/train/exp36/results.csv', file_5l='./runs/train/exp38/results.csv',\
 file_5x='./runs/train/exp40/results.csv' ):
    # Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    save_dir_5s = Path(file_5s).parent
    files_5s = list(save_dir_5s.glob('results*.csv'))
    assert len(files_5s), f'No results.csv files found in {save_dir_5s.resolve()}, nothing to plot.'
########
    save_dir_5m = Path(file_5m).parent
    files_5m = list(save_dir_5m.glob('results*.csv'))
    assert len(files_5m), f'No results.csv files found in {save_dir_5m.resolve()}, nothing to plot.'
##########
    save_dir_5l = Path(file_5l).parent
    files_5l = list(save_dir_5l.glob('results*.csv'))
    assert len(files_5l), f'No results.csv files found in {save_dir_5l.resolve()}, nothing to plot.'
###########
    save_dir_5x = Path(file_5x).parent
    files_5x = list(save_dir_5x.glob('results*.csv'))
    assert len(files_5x), f'No results.csv files found in {save_dir_5x.resolve()}, nothing to plot.'
############
    for fi, f in enumerate(files_5s):
        try:
            data = pd.read_csv(f)
            #s = [x.strip() for x in data.columns]
            x = data.values[:500, 0]
            for i, j in enumerate([6, 7]):
                #for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:500, j]
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker='.', label="YOLOv5s", linewidth=1, markersize=2)
                #ax[i].set_title(s[j], fontsize=12)
                ax[0].set_title("AP_50", fontsize=12)
                ax[1].set_title("AP_50:95", fontsize=12)
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')        
    for fi, f in enumerate(files_5m):
        try:
            data = pd.read_csv(f)
            #s = [x.strip() for x in data.columns]
            x = data.values[:500, 0]
            for i, j in enumerate([6, 7]):
                #for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:500, j]
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker='.', label="YOLOv5m", linewidth=1, markersize=2)
                #ax[i].set_title(s[j], fontsize=12)
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')        
    for fi, f in enumerate(files_5l):
        try:
            data = pd.read_csv(f)
            #s = [x.strip() for x in data.columns]
            x = data.values[:500, 0]
            for i, j in enumerate([6, 7]):
                #for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:500, j]
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker='.', label="YOLOv5l", linewidth=1, markersize=2)
                #ax[i].set_title(s[j], fontsize=12)
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')
    for fi, f in enumerate(files_5x):
        try:
            data = pd.read_csv(f)
            #s = [x.strip() for x in data.columns]
            x = data.values[:500, 0]
            for i, j in enumerate([6, 7]):
                #for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:500, j]
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker='.', label="YOLOv5x", linewidth=1, markersize=2)
                #ax[i].set_title(s[j], fontsize=12)
                                                                                        
              
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')
    ax[0].legend(loc='lower right')
    ax[1].legend(loc='lower right')
    fig.savefig("./plot_train/results.png", dpi=200)
    #plt.imshow(fig)
    plt.close()
plot_results()