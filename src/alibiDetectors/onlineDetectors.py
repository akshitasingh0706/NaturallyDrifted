'''
online (calibrated gradual) drifts on text data from the following detectors - MMD and LSDD
'''

from typing import Callable, Dict, Optional, Union
import nlp
import pandas as pd
import numpy as np
import scipy
import os
import tensorflow as tf
from transformers import AutoTokenizer
from functools import partial
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from alibi_detect.cd import MMDDriftOnline, LSDDDriftOnline
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.cd.tensorflow import UAE
from alibi_detect.cd.tensorflow import preprocess_drift

from sampling import samplingData
from base import detectorParent
class onlineDetectors(samplingData, detectorParent):
    def __init__(self, *args, **kwargs):           
        super(onlineDetectors, self).__init__(*args, **kwargs)
        """
        [description]

        Returns
        ----------  
        [finish]
        """

    def sampleData(self):
        if self.sample_dict is None:
            return samplingData.samples(self)
        else:
            return self.sample_dict

    def preprocess(self):
        sample_dict = self.sampleData()
        data_ref = sample_dict[0]

        layers = [-_ for _ in range(1, self.n_layers + 1)]

        embedding = TransformerEmbedding(self.SBERT_model, self.emb_type, layers)
        tokens = self.tokenizer(list(data_ref[:self.tokenizer_size]), pad_to_max_length=True, 
                                max_length= self.max_len, return_tensors='tf')
        x_emb = embedding(tokens)
        shape = (x_emb.shape[1],)
        uae = UAE(input_layer=embedding, shape=shape, enc_dim= self.enc_dim)
        return uae
    
    def detector(self):
        if self.sample_dict:
            data_ref = self.sample_dict[0]
        else:
            sample_dict = self.sampleData()
            data_ref = sample_dict[0]
        
        uae = self.preprocess()
        preprocess_fn = partial(preprocess_drift, model= uae, tokenizer= self.tokenizer, 
                        max_len= self.max_len, batch_size= self.batch_size)
        if self.test == "MMD": 
            cd = MMDDriftOnline(data_ref, ert = self.ert, window_size = self.window_size, 
                        preprocess_fn=preprocess_fn, n_bootstraps = self.n_bootstraps,
                        device = self.device, input_shape=(self.max_len,))
        elif self.test == "LSDD":
            cd = LSDDDriftOnline(data_ref, ert = self.ert, window_size = self.window_size, 
                        preprocess_fn=preprocess_fn, n_bootstraps = self.n_bootstraps,
                        device = self.device, input_shape=(self.max_len,))
        elif self.test == "LearnedKernel":
            pass
        else:
            print("The following detector is not included in the package yet")
        return cd if self.test in ['MMD', 'LSDD', 'LearnedKernel'] else 0
    
    def run(self):
        if self.sample_dict:
            data_h0 = self.sample_dict[1]
            data_h1 = self.sample_dict[2]
        else:
            sample_dict = self.sampleData()
            data_h0 = sample_dict[1]
            data_h1 = sample_dict[2]
        
        cd = self.detector()
        def time_run(cd, X, window_size):
            n = len(X)
            perm = np.random.permutation(n)
            t = 0
            cd.reset()
            while True:
                pred = cd.predict(X[perm[t%n]])
                if pred['data']['is_drift'] == 1:
                    return t
                else:
                    t += 1
        
        def plot(cd, times_list, ert):
          # get intersection of t-stat overtaking the threshold
          ts = np.arange(cd.t)
          plt.plot(ts, cd.test_stats, label='Test statistic')
          plt.plot(ts, cd.thresholds, label='Thresholds')
          plt.title('Test Statistic and Threshold intersection')
          plt.xlabel('Time window (t)', fontsize=16)
          plt.ylabel('Test Staistics at t ($T_t$)', fontsize=16)
          plt.legend(loc='upper right', fontsize=14)
          plt.show()

          # get the probability plot of inverse erts against geometric distribution
          scipy.stats.probplot(np.array(times_list), dist=scipy.stats.geom, sparams=1/ert, plot=plt)
          plt.show()

        print("No Drift Scenario")
        times_h0 = [time_run(cd, data_h0, self.window_size) for _ in range(self.n_runs)]
        print(f"Average run-time under no-drift: {np.mean(times_h0)}")
        plot(cd, times_h0, self.ert)

        print("Possible Drift Scenario")
        times_h1 = [time_run(cd, data_h1, self.window_size) for _ in range(self.n_runs)]
        print(f"Average run-time under possible-drift: {np.mean(times_h1)}")
        plot(cd, times_h1, self.ert)
            
        times_dict = {}
        times_dict[0] = times_h0   
        times_dict[1] = times_h1   
        return times_dict