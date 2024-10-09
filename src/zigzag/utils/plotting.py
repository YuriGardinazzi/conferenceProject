#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:49:21 2024

@author: karthikviswanathan
"""
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def plot2D(arr, ax, title):
    """
    Example usage -
    data = np.load('lenet/temp5.npz')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    plot2D(data['p0'], ax1, 'PH0')
    plot2D(data['p1'], ax2, 'PH1')
    fig.suptitle('Frequency of persistence pairs')
    plt.tight_layout()
    plt.show()    
    """
    arr = [tuple(item) for item in arr]
    # arr = np.array([arr[:, 0], arr[:, 1] - arr[:, 0]]).T
    pair_counts = dict(Counter(arr))

    # Prepare data for plotting
    pairs, frequencies = zip(*pair_counts.items())
    x, y = zip(*pairs)
    
    # Create a 2D array for frequencies
    # freq_array = np.zeros((max(y)+1, max(x)+1))
    
    freq_array = np.zeros((17, 16))
    for pair, freq in zip(pairs, frequencies):
        freq_array[pair[1]][pair[0]] = np.log10(1 + freq)
        # print(pair[0], pair[1])

    # print(freq_array[-1]) 
    # Create heatmap
    # ax.figure(figsize=(10, 6))
    im = ax.imshow(freq_array, cmap='hot', interpolation='nearest', origin = 'lower')
    fig.colorbar(im, label='Log Frequency', ax = ax)
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(title)

    return freq_array



