from sentence_transformers import SentenceTransformer
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

from base import detectorParent
from sampling import samplingData
from basicDetectors import basicDetectors
from onlineDetectors import onlineDetectors
from contextDetectors import contextDetectors

class alibiDetectors(detectorParent):
    def __init__(self, *args, **kwargs):
        """
        This is final wrapper class for all text related Alibi Detectors (basic detectors, online
        detectors, context aware detectors etc.). We can generally just directly call this one function
        and populate all the parameters (datasets, detection tests, drift types etc.) and get our test
        statistics. 

        Args
        ----------
        data_ref : np.ndarray, list
            Dataset on which the original model is trained (ex: training dataset). We flag a drift 
            with a reference to the distribution of this dataset. 

        data_h0 :  np.ndarray, list 
            This is an optional dataset that we can use as a sanity check for the efficacy of a drift
            detector. Generally, we use the same dataset as data_ref (or a stream that comes soon after).
            The lack of drift in data_h0 (with data_ref as our reference) is the necessary 
            condition to decide the robustness of the drift detection method

        data_h1: np.ndarray, list
            This is the principal dataset on which we might see a drift (ex. deployment data). 
            It can be just one sample (for sudden drifts) or stream of samples (for gradual drifts).
            Often, for pipelines, datasets come in batches, and each new batch can then be updated
            to the new data_h1.
        
        sample_dict: dict
            Dictionary with samples for reference and comparison data (or streams of comparison data).
            The user can directly input the dictionary as our dataset source if they would prefer to 
            organize the data on their own. 

        sample_size: int
            This parameter decides the number of samples from each of the above 3 datasets that we would
            like to work with. For instance, if the entire training data is 100K sentences, we can use
            a sample_size = 500 to randomly sample 500 of those sentences. 
        
        test: str
            Here, we specify the kind of drift detection test we want (KS, KLD, JSD, MMD, LSDD).
            Each of them is described in greater detail in the README.md.  

        drift_type: str
            Drifts can vary depending on the time horizan and frequency at which we try to detect
            them. This parameter asks the user to specify the type of drift ("Sudden", "Gradual", 
            "Online"). The details of each are in README.md
        
        plot: bool
            This parameter asks the user if they wish to see some of the plots of the results
            from the drift detection. Not every detector will result in relevant plot.

        windows: int 
            This parameter is relevant for gradual drifts and helps break down the data into a 
            certain number of buckets. These buckets can act like "batches" or "data streams".
            The idea behind this approach is that we are trying to localize drifts to a certain 
            time frame and check for consistencies (or lack thereof) in detection. 
            If data_h1 has 100K data points, and if we wish to detect drifts
            gradually over time, a proxy approach would be to break the data in sets of 5K points
            and then randomly sample from each set separately. 
        
        SBERT_model: str 
            This parameter is specific to the SBERT embedding models. If we choose to work with SBERT,
            we can specify the type of SBERT embedding out here. Ex. 'bert-base-uncased'  

        ert: int (optional)
            Expected Run Time before a drift is detected. Alibi detect uses this approach for it's 
            online drift detectors. If the average ERT for the reference data is significantly higher
            than the average run time for the drifted data, that might indicate a possible drift. 
        
        window_size: int
            This parameter is used within Alibi's online detectors. 
            It specifies the number of datapoints to include in one window.

        n_run: int
            This parameter is used within Alibi's online detectors and specifies the number of runs
            the detector must perform before we can get an average ERT. 

        n_bootstraps: int
             This parameter is used within Alibi's online detectors

        context_type: str
            Context that we wish to ignore
            1) sub-population: if we wish to ignore the relative change in sub-population of certain 
            classes

        Returns
        ----------  
        Lists and plots of relevant test statistics (p-values, distances) given the selected 
        detector (MMD, LSDD etc) and drift type (Sudden, Gradual, Online)
        """                   
        super(alibiDetectors, self).__init__(*args, **kwargs)
    def run(self):
        if self.test in ["MMD", "LSDD", 'all'] and self.drift_type in ['Sudden', 'Gradual']:
            cd = basicDetectors(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1,
                               sample_size = self.sample_size, windows = self.windows,
                                
                                test = self.test, drift_type = self.drift_type, SBERT_model = self.SBERT_model,
                                transformation = self.transformation, pval_thresh = self.pval_thresh, 
                                dist_thresh = self.dist_thresh,
                                
                                emb_type = self.emb_type, n_layers = self.n_layers, enc_dim = self.enc_dim,
                                tokenizer_size = self.tokenizer_size, batch_size = self.batch_size, max_len = self.max_len
                               )
            cd.run()            
        elif self.test in ["MMD", "LSDD"] and self.drift_type == "Online":
            cd = onlineDetectors(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1,
                               sample_size = self.sample_size, windows = self.windows,
                                
                                test = self.test, drift_type = self.drift_type, SBERT_model = self.SBERT_model,
                                transformation = self.transformation, pval_thresh = self.pval_thresh, 
                                dist_thresh = self.dist_thresh,
                                
                                emb_type = self.emb_type, n_layers = self.n_layers, enc_dim = self.enc_dim,
                                tokenizer_size = self.tokenizer_size, batch_size = self.batch_size, max_len = self.max_len,
                                 
                                 ert = self.ert, n_runs = self.n_runs, window_size = self.window_size
                                )
            cd.run()
        elif self.context is not None:
            cd = contextDetectors(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1)
            cd.run() 
        else:
            print("Please check arguments")

