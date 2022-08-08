import imp
from typing import Callable, Dict, Optional, Union
import numpy as np
from sklearn.neighbors import KernelDensity

from base import detectorParent
from sampling import samplingData
from baseModels import baseModels
from embedding import embedding
from base import detectorParent

class distributions(embedding, samplingData, detectorParent):
    # def __init__(self, bandwidth = .05, kernel = 'gaussian', *args, **kwargs):
    def __init__(self, *args, **kwargs):
        detectorParent.__init__(self, *args, **kwargs)
        samplingData.__init__(self, *args, **kwargs)
        """
        In this class, we construct distributions out of the embeddings we got from the "embedding" class. 
        This is an optional class and is only required if we are running a distribution dependent test such 
        as KLD or JSD. 

        Returns
        ----------  
        A dictionary containing the distributions as decided by the choice of embedding model and 
        drift detection test type
        """
        # self.bandwidth = bandwidth
        # self.kernel = kernel

    def kde(self): # ex. (bandwidth = .05, kernel = 'gaussian')
        return KernelDensity(bandwidth = .05, kernel = 'gaussian')

    def distributions_doc2vec(self): 
        """
        Constructs distributions for Doc2Vec embeddings 

        Returns
        ----------  
          a dictionary containing the distributions as decided by the choice of embedding model and 
        drift detection test type
        """  
        kde = self.kde()
        # distributions across all iterations
        distributions_across_iters = {}
        for it in range(self.iterations):
            # if self.emb_dict is None:
            emb_dict = embedding.final_embeddings(self)
            '''
            distributions for each data window in 1 iteration
            ex. for sudden drift we will only have 3 distributions - ref, h0, h1
            for gradual drifts, we will have distributions for each time window
            '''
            distributions_per_window = {} # distribution per data window
            for ww in emb_dict.keys():  # for each data window (keys)
                data = np.atleast_2d(emb_dict[ww]).T
                kde.fit(data)
                kde_score = kde.score_samples(data)
                distributions_per_window[ww] = kde_score
            distributions_across_iters[it] = distributions_per_window
        return distributions_across_iters

    def distributions_seneconders(self):
        """
        Constructs distributions for SBERT or USE embeddings 

        Returns
        ----------  
        a dictionary containing the distributions as decided by the choice of embedding model and 
        drift dete
        """
        embs = embedding(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, 
                        sample_dict = self.sample_dict, test = self.test, 
                        sample_size = self.sample_size, windows = self.windows, drift_type = self.drift_type, 
                        embedding_model = self.embedding_model, SBERT_model = self.SBERT_model, 
                        transformation = self.transformation)
        kde = KernelDensity(bandwidth = .05, kernel = 'gaussian')
        distributions_across_iters = {}
        for it in range(self.iterations):
            # if self.emb_dict is None:
            emb_dict = embedding.final_embeddings(self)
            distributions_per_window = {}          
            for ww in emb_dict.keys(): # for each data window (keys)
                dimensions = emb_dict[0].shape[1] # ex. if we chose PCA with n_comp = 25, then dimensions = 25
                sent_size = emb_dict[0].shape[0] # generally, sample_size
                '''
                for each dimension in that data window 
                ex. dimensions = 768 if the SBERT_model = bert-base-uncased 
                which goes through no encoding 
                And if say it goes through PCA with n_components = 40, 
                then the dimensions reduce to 40
                '''
                distributions_per_dim = np.zeros((sent_size, dimensions))
                for sent_idx in range(sent_size):
                    data = np.atleast_2d(emb_dict[ww][sent_idx, :]).T
                    kde.fit(data)
                    kde_score = kde.score_samples(data)
                    distributions_per_dim[sent_idx, :] = kde_score               
                distributions_per_window[ww] = distributions_per_dim
            distributions_across_iters[it] = distributions_per_window
        return distributions_across_iters
    
    def final_distributions(self):
        """
        Constructs distributions for the selected embeddings 

        Returns
        ----------  
        a dictionary containing the distributions as decided by the choice of embedding model and 
        drift dete
        """
        if self.embedding_model in ["SBERT", "USE"]:
            return self.distributions_seneconders()
        elif self.embedding_model == "Doc2Vec":
            return self.distributions_doc2vec()
        else:
            print("The specified embedding model is not yet included in the package. We will look into it.")