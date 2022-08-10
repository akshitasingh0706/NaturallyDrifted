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

class allDetectors(detectorParent):
    def __init__(self, *args, **kwargs):           
        super(allDetectors, self).__init__(*args, **kwargs)
    def run(self):
        if self.test in ["MMD", "LSDD"] and self.drift_type in ['Sudden', 'Gradual']:
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

# class allDetectors(onlineDetectors, contextDetectors, basicDetectors):
#     def run(self):
#         if self.test in ["MMD", "LSDD"] and self.drift_type in ['Sudden', 'Gradual']:
#             return basicDetectors.run(self)
#         elif self.test in ["MMD", "LSDD"] and self.drift_type == "Online":
#             return onlineDetectors.run(self)
#         elif self.test in ["MMD", "LSDD"] and self.context_type is not None:
#             return contextDetectors.run(self)
#         else:
#             print("Please look into the test or drift type")


