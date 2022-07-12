from typing import Callable, Dict, Optional, Union
import nlp
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from transformers import AutoTokenizer
from functools import partial

from alibi_detect.cd import KSDrift, MMDDrift, LearnedKernelDrift, ClassifierDrift, LSDDDrift
from alibi_detect.utils.saving import save_detector, load_detector
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.cd.tensorflow import UAE
from alibi_detect.cd.tensorflow import preprocess_drift

from sampling import samplingData

class AlibiDetectors:
    def __init__(self, 
                data_ref: Optional[Union[np.ndarray, list, None]],
                data_h0: Optional[Union[np.ndarray, list, None]],
                data_h1: Union[np.ndarray, list], # "data" in sample_data_gradual
                test: Union["MMDDrift", "LSDDDrift", "LearnedKernel"],
                sample_size: int, 
                windows: Optional[int],
                drift_type: Optional[Union["Sudden", "Gradual"]],
                SBERT_model: str,   
                 
                emb_type: Optional[Union['pooler_output', 'last_hidden_state', 'hidden_state', 'hidden_state_cls']],
                n_layers: Optional[int],
                max_len: Optional[int],
                enc_dim: Optional[int],
                batch_size: Optional[int],
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
            This is the dataset on which is used as the reference/baseline when detecting drifts. 
            For instance, if our test of choice is KL Divergence, then we will declare a possible
            drift based on whether any other data is close in distribution to data_ref. 
            Generally, the goal is to have all future datasets be as close (in embeddings, distributions)
            to data_ref, which is how we conclude that there is no drift in the dataset.  
            
            data_ref is typically sampled from the "training data". During real world application, 
            this is the data on which the test will be modeled on because this would generally be 
            the only data the user would have access to at that point of time. 

        data_h0 :  np.ndarray, list (optional)
            This is generally the same dataset as data_ref (or a stream that comes soon after).
            We use the lack of drift in data_h0 (with data_ref as our reference) as the necessary 
            condition to decide the robustness of the drift detection method. If the method ends up 
            detecting a drift in data_h0 itself, we know it is most likely not doing a good job. 
            This is because both data_ref and data_h0 are expected to be coming from the same source 
            and hence should result in similar embeddings and distributions. 

            If the user is confident in the efficacy of their drift detection method, then it would be 
            worthwhile to consider change the size of data_ref and data_h0 and then re-evaluate detector
            performance, before proceeding to data_h1. 

        data_h1: np.ndarray, list
            This is the primary dataset on which we can expect to possibly detect a drift. In the real 
            world, this would usually be the dataset we get post model deployment. To test detectors, a
            convenient (but not necessarily the best) practice is to take the test data and use that as
            our proxy for the deployed dataset. 

            Multiple research papers and libraries tend to also use "perturbed" data for their choice of
            data_h1. Perturbations can include corruptions in images (vision data) or introduction of 
            unneccessary words and phrases (text data). This is generally the first step in testing the 
            efficacy of a drift detection method. Once again, if the detectors fails to detect a drift
            on manually perturbed data, then its quite likely it will not be able to detect drifts in 
            the real, deployed data as well. 

            Therefore, for our purposes, we have tried to minimize the use of artifically perturbed data
            and instead rely on test data/data from far away time periods as our data_h1 source. 

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

        windows: int (optional)
            This decided the number of segments we would like to break the data into. 
            This parameter is only required for gradual/incremental drift detection. 
            For instance, if data_h1 has 100K data points, and if we wish to detect drifts
            gradually over time, a proxy approach would be to break the data in sets of 5K points
            and then randomly sample from each set separately. 
        
        SBERT_model: str
            This parameter is specific to the SBERT embedding models. If we choose to work with SBERT,
            we can specify the type of SBERT embedding out here. Ex. 'bert-base-uncased'  

        Returns
        ----------  
        A dictionary containing the distributions as decided by the choice of embedding model and 
        drift detection test type
        """

        self.data_ref = data_ref
        self.data_h0  = data_h0
        self.data_h1  = data_h1
        self.test = test
        self.sample_size = sample_size
        self.windows = windows
        self.drift_type = drift_type
        self.SBERT_model = SBERT_model

        self.emb_type = emb_type
        self.n_layers = n_layers
        self.max_len = max_len
        self.enc_dim = enc_dim
        self.tokenizer_size = tokenizer_size
        self.tokenizer = AutoTokenizer.from_pretrained(SBERT_model)
        self.batch_size = batch_size

    def preprocess(self):
        layers = [-_ for _ in range(1, self.n_layers + 1)]

        embedding = TransformerEmbedding(self.SBERT_model, self.emb_type, layers)
        tokens = self.tokenizer(list(self.data_ref[:self.tokenizer_size]), pad_to_max_length=True, 
                                max_length= self.max_len, return_tensors='tf')
        x_emb = embedding(tokens)
        shape = (x_emb.shape[1],)
        uae = UAE(input_layer=embedding, shape=shape, enc_dim= self.enc_dim)
        return uae

    def detector(self):
        sample = samplingData(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, 
                               drift_type = self.drift_type, sample_size = self.sample_size, windows = self.windows)
        sample_dict = sample.samples()
        data_ref = sample_dict[0]

        uae = self.preprocess()
        preprocess_fn = partial(preprocess_drift, model= uae, tokenizer= self.tokenizer, 
                        max_len= self.max_len, batch_size= self.batch_size)
        if self.test == "MMDDrift": 
            cd = MMDDrift(data_ref, p_val=.05, preprocess_fn=preprocess_fn, 
                      n_permutations=100, input_shape=(self.max_len,))
        elif self.test == "LSDDDrift":
            cd = LSDDDrift(data_ref, p_val=.05, preprocess_fn=preprocess_fn, 
                      n_permutations=100, input_shape=(self.max_len,))
        elif self.test == "LearnedKernel":
            pass
        else:
            print("The following is not ")
        return cd 
    
    def predict(self):
        labels = ['No!', 'Yes!']
        cd = self.detector()
  
        sample = samplingData(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, 
                               drift_type = self.drift_type, sample_size = self.sample_size, windows = self.windows)
        sample_dict = sample.samples()

        if self.drift_type == "Sudden":  
            for i, data_name in enumerate(["X_h0", "X_comp"]):
                data = sample_dict[i+1]
                print("Drift results for ", data_name ,"data using ", self.test, "test:")
                preds = cd.predict(data)
                print('Drift? {}'.format(labels[preds['data']['is_drift']]))
                print('p-value: {}'.format(preds['data']['p_val']))

        elif self.drift_type == "Gradual":
            if type(self.windows) is not int:
                print("Please fill/correct windows parameter")
            for ww in range(1,self.windows):
                data = sample_dict[ww]
                print("Drift results for window: ", ww, "data using", self.test, "test:")
                preds = cd.predict(data)
                print('Drift? {}'.format(labels[preds['data']['is_drift']]))
                print('p-value: {}'.format(preds['data']['p_val']))
        else:
            print("The following drift type is not included")
