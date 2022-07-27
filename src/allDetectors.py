from math import log2
import numpy as np
from scipy import stats
import pandas as pd
from typing import Callable, Dict, Optional, Union
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from transformers import AutoTokenizer
from functools import partial
import nlp

from alibi_detect.cd import KSDrift, MMDDrift, LearnedKernelDrift, ClassifierDrift, LSDDDrift
from alibi_detect.utils.saving import save_detector, load_detector
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.cd.tensorflow import UAE
from alibi_detect.cd.tensorflow import preprocess_drift

from sampling import samplingData
from baseModels import baseModels
from embedding import embedding
from distributions import distributions
from myDetectors import myDetectors
from alibiDetectors import alibiDetectors

class allDetectors:
    def __init__(self, 
                data_ref: Optional[Union[np.ndarray, list, None]] = None,
                data_h0: Optional[Union[np.ndarray, list, None]] = None,
                data_h1: Optional[Union[np.ndarray, list]] = None, # "data" in sample_data_gradual
                sample_dict: Optional[Dict] = None,
                test: Union["KS", "KL", "JS", "MMDDrift", "LSDDDrift", "LearnedKernel"] = "MMDDrift",
                pval_thresh = .05,
                dist_thresh: int = .0009,

                emb_type: Optional[Union['pooler_output', 'last_hidden_state', 'hidden_state', 'hidden_state_cls']] = 'hidden_state',
                n_layers: Optional[int] = 9,
                max_len: Optional[int] = 200,
                enc_dim: Optional[int] = 32,
                batch_size: Optional[int] = 32,
                tokenizer_size: Optional[int] = 3, # keep it small else computer breaks down             
                
                sample_size: int = 500, 
                iterations: int = 20,
                windows: Optional[int] = 20,
                drift_type: Optional[Union["Sudden", "Gradual"]] = "Sudden",
                embedding_model: Union["Doc2Vec", "SBERT", "USE"] = "Doc2Vec",
                SBERT_model: Optional[Union[str, None]] = None, 
                transformation: Union["PCA", "SVD", "UMAP", None] = None,
                ):
        """
        This class returns the final detection results based on the embeddings or distributions it
        inherits. Currently, the tests covered in this class are Kolmogorov–Smirnov test, 
        Kullback–Leibler divergence, and Jensen-Shannon Divergence. Each test will return a different 
        output based on the kind of embedding model we choose to work with. 
        
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

        windows: int (optional)
            This decided the number of segments we would like to break the data into. 
            This parameter is only required for gradual/incremental drift detection. 
            For instance, if data_h1 has 100K data points, and if we wish to detect drifts
            gradually over time, a proxy approach would be to break the data in sets of 5K points
            and then randomly sample from each set separately. 
        
        embedding_model: str
            This is the principle parameter of this class. It decided the kind of embedding the text 
            goes through. The embeddings we consider thus far are: 
            a) SBERT: A Python framework for state-of-the-art sentence, text and image embeddings. 
            b) Universal Sentence Encoders: USE encodes text into high dimensional vectors that can be 
            used for text classification, semantic similarity, clustering, and other natural language tasks
            c) Doc2Vec: a generalization of Word2Vec, which in turn is an algorithm that uses a 
            neural network model to learn word associations from a large corpus of text
        
        SBERT_model: str
            This parameter is specific to the SBERT embedding models. If we choose to work with SBERT,
            we can specify the type of SBERT embedding out here. Ex. 'bert-base-uncased'
        
        transformation: str (optional)
            Embeddings render multiple multi-dimensional vector spaces. For instance, USE results in 512 
            dimensions, and 'bert-base-uncased' results in 768 dimensions. For feature levels tests such 
            as KLD or JSD, such a large dimension might not be feasible to analyse, and thus we can reduce 
            the dimensionality by selecting the most important components using methods such as PCA and SVD.

        iterations: int
            We can run through multiple iterations of the embeddings to make our drift detection test more
            robust. For instance, if we only detect a drift on 1 out of 10 itertions, then we might be 
            better off not flagging a drift at all.  
        """

        self.data_ref = data_ref
        self.data_h0  = data_h0
        self.data_h1  = data_h1
        self.sample_dict = sample_dict
        self.test = test
        self.sample_size = sample_size
        self.windows = windows
        self.drift_type = drift_type
        self.embedding_model = embedding_model
        self.SBERT_model = SBERT_model
        self.transformation = transformation
        self.pval_thresh = pval_thresh
        self.dist_thresh = dist_thresh
        self.iterations = iterations

        self.emb_type = emb_type
        self.n_layers = n_layers
        self.max_len = max_len
        self.enc_dim = enc_dim
        self.tokenizer_size = tokenizer_size
        self.tokenizer = AutoTokenizer.from_pretrained(SBERT_model) if SBERT_model else None
        self.batch_size = batch_size
    
    def run(self):
        if self.test in ['KL', 'JS', 'KS']:
            myDets = myDetectors(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, 
                sample_dict = self.sample_dict, test = self.test, 
                sample_size = self.sample_size, windows = self.windows, drift_type = self.drift_type, 
                embedding_model = self.embedding_model, SBERT_model = self.SBERT_model, 
                transformation = self.transformation, iterations = self.iterations,
                pval_thresh = self.pval_thresh, dist_thresh = self.dist_thresh)
            myDets.run()
        else:
            alibiDets = alibiDetectors(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, 
                sample_dict = self.sample_dict, test = self.test, sample_size = self.sample_size, 
                drift_type = self.drift_type, SBERT_model = self.SBERT_model, 
                transformation = self.transformation, windows = self.windows,
                emb_type = self.emb_type, n_layers = self.n_layers, max_len = self.max_len, 
                enc_dim = self.enc_dim, batch_size = self.batch_size, tokenizer_size = self.tokenizer_size,
                pval_thresh = self.pval_thresh, dist_thresh = self.dist_thresh)
            alibiDets.run()
    


