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
                model_name: str,   
                 
                # emb_type: Optional[str],
                emb_type: Optional[Union['pooler_output', 'last_hidden_state', 'hidden_state', 'hidden_state_cls']],
                n_layers: Optional[int],
                max_len: Optional[int],
                enc_dim: Optional[int],
                batch_size: Optional[int],
                tokenizer_size: Optional[int] = 3 # keep it small else computer breaks down
               
                 ):
        self.data_ref = data_ref
        self.data_h0  = data_h0
        self.data_h1  = data_h1
        self.test = test
        self.sample_size = sample_size
        self.windows = windows
        self.drift_type = drift_type
        self.model_name = model_name

        self.emb_type = emb_type
        self.n_layers = n_layers
        self.max_len = max_len
        self.enc_dim = enc_dim
        self.tokenizer_size = tokenizer_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size

    def preprocess(self):
        layers = [-_ for _ in range(1, self.n_layers + 1)]

        embedding = TransformerEmbedding(self.model_name, self.emb_type, layers)
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
