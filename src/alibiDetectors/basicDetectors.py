'''
sudden and (basic) gradual drifts on text data from the following detectors - MMD and LSDD
'''

from typing import Callable, Dict, Optional, Union
import nlp
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from transformers import AutoTokenizer
from functools import partial
import matplotlib.pyplot as plt

from alibi_detect.cd import KSDrift, MMDDrift, LearnedKernelDrift, ClassifierDrift, LSDDDrift
from alibi_detect.utils.saving import save_detector, load_detector
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.cd.tensorflow import UAE
from alibi_detect.cd.tensorflow import preprocess_drift

from sampling import samplingData
from base import detectorParent

class basicDetectors(samplingData, detectorParent):
    def __init__(self, *args, **kwargs):
        detectorParent.__init__(self, *args, **kwargs)
        samplingData.__init__(self, *args, **kwargs)
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
            cd = MMDDrift(data_ref, p_val=.05, preprocess_fn=preprocess_fn, 
                      n_permutations=100, input_shape=(self.max_len,))
        elif self.test == "LSDD":
            cd = LSDDDrift(data_ref, p_val=.05, preprocess_fn=preprocess_fn, 
                      n_permutations=100, input_shape=(self.max_len,))
        elif self.test == "LearnedKernel":
            pass
        else:
            print("The following detector is not included in the package yet")
        return cd if self.test in ['MMD', 'LSDD', 'LearnedKernel'] else 0
    
    def run(self):
        labels = ['No!', 'Yes!']
        cd = self.detector()

        sample_dict = self.sampleData()
        pvalues = []
        distances = []
        if self.drift_type == "Sudden":  
            for i, data_name in enumerate(["X_h0", "X_comp"]):
                data = sample_dict[i+1]
                print("Drift results for ", data_name ,"data using ", self.test, "test:")
                preds = cd.predict(data)
                print('Drift? {}'.format(labels[preds['data']['is_drift']]))
                print('p-value: {}'.format(preds['data']['p_val']))
                pvalues.append(preds['data']['p_val'])
                distances.append(preds['data']['distance'])

        elif self.drift_type == "Gradual":
            for ww in range(1,len(sample_dict)):
                data = sample_dict[ww]
                print("Drift results for window: ", ww, "data using", self.test, "test:")
                preds = cd.predict(data)
                print('Drift? {}'.format(labels[preds['data']['is_drift']]))
                print('p-value: {}'.format(preds['data']['p_val']))
                pvalues.append(preds['data']['p_val'])
                distances.append(preds['data']['distance'])

            if self.plot:
                plt.plot(pvalues)
                plt.title("P-Values for Non-Calibrated Gradual Data broken by Time-Windows")
                plt.xlabel("Time Windows")
                plt.xlabel("P-values")                
                plt.show()
                plt.plot(distances)
                plt.title("Distances for Non-Calibrated Gradual Data broken by Time-Windows")
                plt.xlabel("Time Windows")
                plt.xlabel("Distances") 
                plt.show()

        else:
            print("The following drift type is not included")

        test_stats = {}
        test_stats['pvalues'] = pvalues
        test_stats['distances'] = distances

        return test_stats
