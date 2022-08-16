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
import seaborn as sns

from alibi_detect.cd import KSDrift, MMDDrift, LearnedKernelDrift, ClassifierDrift, LSDDDrift
from alibi_detect.utils.saving import save_detector, load_detector
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.cd.tensorflow import UAE
from alibi_detect.cd.tensorflow import preprocess_drift

from sampling import samplingData
from base import detectorParent

class basicDetectors(samplingData, detectorParent):
    def __init__(self, *args, **kwargs):
        """
        In this class, we check for possible sudden drift in the data, using some of Alibi's methods.
        Sudden drifts are drifts we could see right after deployment. 
        We can also use sudden drift techniques to try identifying drifts in 
        a new batch of data (Ex. data being streamed weekly). 

        Returns
        ----------  
        Lists and plots of relevant test statistics (p-values, distances) given the selected 
        detector (MMD, LSDD etc)
        """
        super(basicDetectors, self).__init__(*args, **kwargs)

    def sampleData(self):
        """
        Call the samplingData class to construct samples from the input data provided by the user

        Returns
        ----------  
        Dictionary with samples for reference and comparison data (or streams of comparison data).
        """
        if self.sample_dict is None:
            return samplingData.samples(self)
        else:
            return self.sample_dict

    def preprocess(self):
        """
        Here we process the text data in the following manner:
        1) Embed it (generally, by using some kind of a Sentence Transformer)
        2) Prepare a dimension reduction model for it that we can than feed into the main Alibi 
        detector function

        Returns
        ----------  
        A dimesnion reduction/preprocessing model that the Alibi Detector can use (generally, an Untrained Autoencoder)
        """
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
        """
        Here, we call the relevant drift detection method from Alibi Detect, given user input. 
        The function uses reference samples and preprocessing from the previous function as arguments
        for the detection model development here. 

        Returns
        ----------  
        A trained detection model (MMD, LSDD etc) as specified by the user input
        """
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
        """
        Here, we run the detection model from the previous function, on the comparison data on 
        which we want to check for a possible drift. 

        Returns
        ----------  
        Lists and plots of relevant test statistics (p-values, distances) given the selected detector (MMD, LSDD etc)
        """
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
                windows = range(1, self.windows)
                sns.set(rc={'axes.facecolor':'lightblue', 'figure.facecolor':'lightgreen'})
                p = sns.lineplot(x = windows, y = pvalues, markers= 'o', color = 'blue')
                p.axhline(self.pval_thresh, color = 'red', linestyle = '-')
                p.set_xlabel("Time Windows", fontsize = 12, color = 'Blue')
                p.set_ylabel("P-Values", fontsize = 12, color = 'Blue')
                p.set_title(f"P-Values for %s Drift Detector per Data Window  " %self.test 
                            ,fontsize = 13, color = 'Blue')
                plt.show()

                windows = range(1, self.windows)
                sns.set(rc={'axes.facecolor':'lightgreen', 'figure.facecolor':'lightblue'})
                p = sns.lineplot(x = windows, y = distances, markers= 'o', color = 'blue')
                p.axhline(self.dist_thresh, color = 'red', linestyle = '-')
                p.set_xlabel("Time Windows", fontsize = 12, color = 'Blue')
                p.set_ylabel("Distances", fontsize = 12, color = 'Blue')
                p.set_title(f"Distances for %s Drift Detector per Data Window" %self.test  
                            ,fontsize = 13, color = 'Blue')  
                plt.show()         

        else:
            print("The following drift type is not included")

        test_stats = {}
        test_stats['pvalues'] = pvalues
        test_stats['distances'] = distances

        return test_stats
