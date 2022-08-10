'''
Context Aware MMD Detector
'''

from typing import Callable, Dict, Optional, Union
import nlp
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from transformers import AutoTokenizer
from functools import partial
from scipy.special import softmax
import torch.nn as nn
# import umap
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List

from alibi_detect.cd import MMDDrift, ContextMMDDrift
from alibi_detect.utils.saving import save_detector, load_detector
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.cd.tensorflow import UAE
from alibi_detect.cd.tensorflow import preprocess_drift
from alibi_detect.utils.pytorch.data import TorchDataset

from sampling import samplingData
from base import detectorParent

class contextDetectors(samplingData, detectorParent):
    def __init__(self, *args, **kwargs):
        super(contextDetectors, self).__init__(*args, **kwargs)
        """
        [description]

        Returns
        ----------  
        [finish]
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def batch_to_device(self, batch: dict, target_device: torch.device):
        """ Send a pytorch batch to a device (CPU/GPU). """
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(target_device)
        return batch

    def tokenize(self, x: List[str]) -> Dict[str, torch.Tensor]:
        tokens = clf.encode_text.tokenize(x)
        return self.batch_to_device(tokens, self.device)

    def train_model(self, model, loader, epochs=3, lr=1e-3):
        device = self.device
        clf = Classifier().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        for _ in range(epochs):
            for x, y in tqdm(loader):
                tokens, y = self.tokenize(x), y.to(device)
                y_hat = clf(tokens)
                optimizer.zero_grad()
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                
    def eval_model(self, model, loader, verbose=1):
        model.eval()
        logits, labels = [], []
        with torch.no_grad():
            if verbose == 1:
                loader = tqdm(loader)
            for x, y in loader:
                tokens = self.tokenize(x)
                y_hat = model(tokens)
                logits += [y_hat.cpu().numpy()]
                labels += [y.cpu().numpy()]
        logits = np.concatenate(logits, 0)
        preds = np.argmax(logits, 1)
        labels = np.concatenate(labels, 0)
        if verbose == 1:
            accuracy = (preds == labels).mean()
            print(f'Accuracy: {accuracy:.3f}')
        return logits, preds

    def context(self, x: List[str], y: np.ndarray):  # y only needed for the data loader
        """ Condition on classifier prediction probabilities. """
        loader = DataLoader(TorchDataset(x, y), batch_size=32, shuffle=False)
        logits = self.eval_model(clf.eval(), loader, verbose=0)[0]
        return softmax(logits, -1)
    
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
        return cd 
    
    def run(self):


        train_loader = DataLoader(TorchDataset(x_train, y_train), batch_size=32, shuffle=True)
        drift_loader = DataLoader(TorchDataset(x_drift, y_drift), batch_size=32, shuffle=False)

        return 
