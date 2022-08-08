'''
online (calibrated gradual) drifts on text data from the following detectors - MMD and LSDD
'''

from typing import Callable, Dict, Optional, Union
import nlp
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from transformers import AutoTokenizer
from functools import partial

from alibi_detect.cd import MMDDriftOnline, LSDDDriftOnline
from alibi_detect.utils.saving import save_detector, load_detector
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.cd.tensorflow import UAE
from alibi_detect.cd.tensorflow import preprocess_drift

from sampling import samplingData

class onlineDetectors:
    def __init__(self, 
                data_ref: Optional[Union[np.ndarray, list, None]] = None,
                data_h0: Optional[Union[np.ndarray, list, None]] = None,
                data_h1: Optional[Union[np.ndarray, list]] = None, # "data" in sample_data_gradual
                sample_dict: Optional[Dict] = None,

                
                test: Union["MMD", "LSDD"] = "MMD",
                sample_size: int = 500, 
                drift_type: str = "Online",
                SBERT_model: str = 'bert-base-uncased',   
                transformation: Union["UMAP", "UAE", None] = None,
                ert: int = 50,
                window_size: int = 10,
                n_runs: int = 100,
                n_bootraps: Optional[int]= 2500,
                plot: bool = True,
                 
                emb_type: Optional[Union['pooler_output', 'last_hidden_state', 'hidden_state', 'hidden_state_cls']] = 'hidden_state',
                n_layers: Optional[int] = 9,
                max_len: Optional[int] = 200,
                enc_dim: Optional[int] = 32,
                batch_size: Optional[int] = 32,
                tokenizer_size: Optional[int] = 3 # keep it small else computer breaks down 
                 ):
        """
        In this class, we implement some drift detection tests as developed by the Github Repo Alibi Detect.
        Alibi Detect is a large repository with many different kinds of tests for different datasets (tabular,
        vision, text). Here, we try to conveniently integrate some of the tests (MMD, LSDD) specifically 
        for the text setting. 
        
        Args
        ----------
        data_ref : np.ndarray, list
            Dataset on which model is trained (ex: training dataset). We compare a drift with a
            reference to this distribution.

        data_h0 :  np.ndarray, list (optional)
            Generally, the same dataset as data_ref (or a stream that comes soon after).
            We use the lack of drift in data_h0 (with data_ref as our reference) as the necessary 
            condition to decide the robustness of the drift detection method. 

        data_h1: np.ndarray, list
            Principal dataset on which we might see a drift (ex. deployment data). It can be just one
            sample (for sudden drifts) or stream of samples (for gradual drifts)
        
        sample_dict: dict
            Dictionary with samples for reference and comparison data (or streams of comparison data)

        test: str
            Here, we specify the kind of drift detection test we want (KS, KLD, JSD, MMD, LSDD).
            The descriptions for each are discussed in the README.md.  

        sample_size: int
            This parameter decides the number of samples from each of the above 3 datasets that we would
            like to work with. For instance, if the entire training data is 100K sentences, we can use
            a sample_size = 500 to randomly sample 500 of those sentences. 

        drift_type: str
            As discussed in the README, there are many different types of drifts. Here we specify
            the drift type we are looking for, based on the time/frquency. drift_type asks us 
            to specify whether we want to detect sudden drifts or more gradual drifts. 

        SBERT_model: str
            This parameter is specific to the SBERT embedding models. If we choose to work with SBERT,
            we can specify the type of SBERT embedding out here. Ex. 'bert-base-uncased' 

        ert: int
            Expected Run Time before we detect any change

        window_size: int 
            Size of a window 

        Returns
        ----------  
        A dictionary containing the distributions as decided by the choice of embedding model and 
        drift detection test type
        """

        self.data_ref = data_ref
        self.data_h0  = data_h0
        self.data_h1  = data_h1
        self.sample_dict = sample_dict
        self.test = test
        self.sample_size = sample_size
        self.drift_type = drift_type
        self.SBERT_model = SBERT_model
        self.transformation = transformation
        self.ert = ert
        self.window_size = window_size
        self.n_runs = n_runs 
        self.n_bootraps = n_bootraps
        self.plot = plot

        self.emb_type = emb_type
        self.n_layers = n_layers
        self.max_len = max_len
        self.enc_dim = enc_dim
        self.tokenizer_size = tokenizer_size
        self.tokenizer = AutoTokenizer.from_pretrained(SBERT_model)
        self.batch_size = batch_size

    def sampleData(self):
        if self.sample_dict is None:
            sample = samplingData(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, 
                                drift_type = self.drift_type, sample_size = self.sample_size, windows = self.windows)
            return sample.samples()
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
                        p_val=.05, preprocess_fn=preprocess_fn, n_bootstrap = self.n_bootraps,
                        n_permutations=100, input_shape=(self.max_len,))
        elif self.test == "LSDD":
            cd = LSDDDriftOnline(data_ref, ert = self.ert, window_size = self.window_size, 
                        p_val=.05, preprocess_fn=preprocess_fn, n_bootstrap = self.n_bootraps,
                        n_permutations=100, input_shape=(self.max_len,))
        elif self.test == "LearnedKernel":
            pass
        else:
            print("The following detector is not included in the package yet")
        return cd 
    
    def run(self):
        if self.sample_dict:
            data_h0 = self.sample_dict[1]
            data_h1 = self.sample_dict[2]
        else:
            sample_dict = self.sampleData()
            data_h0 = self.sample_dict[1]
            data_h1 = sample_dict[2]
        
        cd = self.detector()

        times_h0 = []
        times_h1 = []
        for _ in range(self.n_runs):
            n_h0 = len(data_h0)
            perm_h0 = np.random.permutation(n_h0)
            time_elapsed = 0
            cd.reset()
            while True:
                pred = cd.predict(data_h1[perm_h0[time_elapsed%n_h0]])
                if pred['data']['is_drift'] == 1:
                    times_h0.append(time_elapsed) 
                else:
                    time_elapsed += 1 

        for _ in range(self.n_runs):
            n_h1 = len(data_h1)
            perm_h1 = np.random.permutation(n_h1)
            time_elapsed = 0
            cd.reset()
            while True:
                pred = cd.predict(data_h1[perm_h1[time_elapsed%n_h1]])
                if pred['data']['is_drift'] == 1:
                    times_h1.append(time_elapsed) 
                else:
                    time_elapsed += 1 
            
        times_dict = {}
        times_dict[0] = times_h0   
        times_dict[1] = times_h1   
        return times_dict    
    