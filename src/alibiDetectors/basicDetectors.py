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
from sentence_transformers import SentenceTransformer
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from alibi_detect.cd import KSDrift, MMDDrift, LearnedKernelDrift, ClassifierDrift, LSDDDrift
from alibi_detect.utils.saving import save_detector, load_detector
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.cd.tensorflow import UAE
from alibi_detect.cd.tensorflow import preprocess_drift
from alibi_detect.utils.pytorch.kernels import DeepKernel

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

    # embed data - for learned kernel
    def embedData(self):
        """
        Call the samplingData class to construct samples from the input data provided by the user

        Returns
        ----------  
        Dictionary with samples for reference and comparison data (or streams of comparison data).
        """
        model = SentenceTransformer(self.SBERT_model)
        temp_dict = self.sampleData()
        sample_dict = {}
        for ww in temp_dict.keys():
            sample_dict[ww] = model.encode(temp_dict[ww])
        return sample_dict
        
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
            sample_dict = self.embedData()
            kernel = DeepKernel(self.proj, eps=0.01)
            cd = LearnedKernelDrift(sample_dict[0], kernel, backend='pytorch', 
                    p_val= self.pval_thresh, epochs=1)
        elif self.test == 'all':
            cd_dict = {}
            # model for MMD drifts
            cd_mmd = MMDDrift(data_ref, p_val=.05, preprocess_fn=preprocess_fn, 
                      n_permutations=100, input_shape=(self.max_len,))
            cd_dict['MMD'] = cd_mmd
            # model for LSDD drifts
            cd_lsdd = LSDDDrift(data_ref, p_val=.05, preprocess_fn=preprocess_fn, 
                      n_permutations=100, input_shape=(self.max_len,))
            cd_dict['LSDD'] = cd_lsdd
            # model for Learned Kernel drifts
            sample_dict = self.embedData()
            kernel = DeepKernel(self.proj, eps=0.01)
            cd_lk = LearnedKernelDrift(sample_dict[0], kernel, backend='pytorch', 
                    p_val= self.pval_thresh, epochs=1)
            cd_dict['LK'] = cd_lk
        else:
            print("The following detector is not included in the package yet")
        return cd if self.test in ['MMD', 'LSDD', 'LearnedKernel'] else cd_dict
    
    # only run if the user wants all detector results
    def run_all(self):
        cd = self.detector() # must be a dictionary by default
        sample_dict = self.sampleData()
        embed_dict = self.embedData()
        test_stats = {'pvals_mmd': [], 'pvals_lsdd': [],  'pvals_lk': [],
                    'dist_mmd': [], 'dist_lsdd': [],  'dist_lk': [] }

        for ww in range(1,len(sample_dict)):
            data = sample_dict[ww]
            embedData = embed_dict[ww] # required for Learned Kernel
            # MMD
            mmd_preds = cd['MMD'].predict(data)
            test_stats['pvals_mmd'].append(mmd_preds['data']['p_val'])
            test_stats['dist_mmd'].append(mmd_preds['data']['distance'])
            # LSDD
            lsdd_preds = cd['LSDD'].predict(data)
            test_stats['pvals_lsdd'].append(lsdd_preds['data']['p_val'])
            test_stats['dist_lsdd'].append(lsdd_preds['data']['distance'])
            # Learned Kernel
            lk_preds = cd['LK'].predict(embedData)
            test_stats['pvals_lk'].append(lk_preds['data']['p_val'])
            test_stats['dist_lk'].append(lk_preds['data']['distance'])

        windows = range(1, self.windows) if self.windows else 2 # either gradual or sudden drifts
        sns.set(rc={'axes.facecolor':'lightblue', 'figure.facecolor':'lightgreen'})
        p = sns.lineplot(x = windows, y = test_stats['pvals_mmd'] , markers= 'o', color = 'blue')
        p = sns.lineplot(x = windows, y = test_stats['pvals_lsdd'] , markers= 'o', color = 'green')
        p = sns.lineplot(x = windows, y = test_stats['pvals_lk'] , markers= 'o', color = 'purple')
        p.axhline(self.pval_thresh, color = 'red', linestyle = '-')
        p.set_xlabel("Time Windows", fontsize = 12, color = 'Blue')
        p.set_ylabel("P-Values", fontsize = 12, color = 'Blue')
        p.set_title("P-Values for All Drift Detector per Data Window  "
                    ,fontsize = 13, color = 'Blue')
        p.legend(labels = ['MMD', 'LSDD', 'Learned Kernel'])
        plt.show()

        sns.set(rc={'axes.facecolor':'lightgreen', 'figure.facecolor':'lightblue'})
        p = sns.lineplot(x = windows, y = test_stats['dist_mmd'] , markers= 'o', color = 'blue')
        p = sns.lineplot(x = windows, y = test_stats['dist_lsdd'] , markers= 'o', color = 'green')
        p = sns.lineplot(x = windows, y = test_stats['dist_lk'] , markers= 'o', color = 'purple')
        p.axhline(self.dist_thresh, color = 'red', linestyle = '-')
        p.set_xlabel("Time Windows", fontsize = 12, color = 'Blue')
        p.set_ylabel("Distances", fontsize = 12, color = 'Blue')
        p.set_title("Distances for All Drift Detector per Data Window"  
                    ,fontsize = 13, color = 'Blue')
        p.legend(labels = ['MMD', 'LSDD', 'Learned Kernel'])
        plt.show()

        return test_stats

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
        
        if self.test in ['MMD', 'LSDD']:
            sample_dict = self.sampleData()
        elif self.test == 'LearnedKernel':
            sample_dict = self.embedData()
        elif self.test == 'all':
            pass
        else:
            print("This test is not included yet")
        
        if isinstance(cd, dict): # (or self.test == 'all')
            return self.run_all()

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
