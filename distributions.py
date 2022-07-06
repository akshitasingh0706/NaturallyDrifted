from typing import Callable, Dict, Optional, Union
import numpy as np
from sklearn.neighbors import KernelDensity

from sampling import samplingData
from baseModels import baseModels
from embedding import embedding

class distributions:
    def __init__(self,
                data_ref: Optional[Union[np.ndarray, list, None]],
                data_h0: Optional[Union[np.ndarray, list, None]],
                data_h1: Union[np.ndarray, list], # "data" in sample_data_gradual
                test : Optional[Union["KS", "KL", "JS"]],
                sample_size: int = 500, 
                windows: int = 20,
                drift_type: Optional[Union["Sudden", "Gradual"]] = "Sudden",
                embedding_model: Union["Doc2Vec", "SBERT"] = "Doc2Vec",
                model_name: Optional[Union[str, None]] = None,
                iterations: int = 500,
                transformation: Optional[Union["PCA", "SVD", "UMAP", "UAE", None]] = None
                #emb_dict: Optional[dict] = None
                 ):
        # self.emb_dict = emb_dict
        self.data_ref = data_ref
        self.data_h0  = data_h0
        self.data_h1  = data_h1
        self.sample_size = sample_size
        self.windows = windows
        self.drift_type = drift_type
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.iterations = iterations
        self.transformation = transformation
        self.test = test

        self.bandwidth = .05
        self.kernel = 'gaussian'

    def kde(self): # ex. (bandwidth = .05, kernel = 'gaussian')
        return KernelDensity(bandwidth = self.bandwidth, kernel = self.kernel)

    def distributions_doc2vec(self):   
        kde = self.kde()
        # distributions across all iterations
        distributions_across_iters = {}
        for it in range(self.iterations):
            # if self.emb_dict is None:
            embs = embedding(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, test = self.test,
                            sample_size = self.sample_size, windows = self.windows, drift_type = self.drift_type, 
                            embedding_model = self.embedding_model, model_name = self.model_name, 
                            transformation = self.transformation, iterations = None)
            emb_dict = embs.final_embeddings()
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

    def distributions_sbert(self):
        embs = embedding(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, test = self.test, 
                        sample_size = self.sample_size, windows = self.windows, drift_type = self.drift_type, 
                        embedding_model = self.embedding_model, model_name = self.model_name, transformation = self.transformation)
        final_dict = embs.final_embeddings()  
        kde = KernelDensity(bandwidth = .05, kernel = 'gaussian')
        distributions_across_iters = {}
        for it in range(self.iterations):
            # if self.emb_dict is None:
            embs = embedding(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1,
                             sample_size = self.sample_size, windows = self.windows, drift_type = self.drift_type, 
                            model_name = self.model_name, transformation = self.transformation, test = self.test,
                            embedding_model = self.embedding_model, iterations = None)
            emb_dict = embs.final_embeddings()
            distributions_per_window = {}          
            for ww in emb_dict.keys(): # for each data window (keys)
                dimensions = emb_dict[0].shape[1] # ex. if we chose PCA with n_comp = 25, then dimensions = 25
                sent_size = emb_dict[0].shape[0] # generally, sample_size
                print("dims", dimensions)
                print("dims", sent_size)
                '''
                for each dimension in that data window 
                ex. dimensions = 768 if the model_name = bert-base-uncased 
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
        if self.embedding_model == "SBERT":
            return self.distributions_sbert()
        elif self.embedding_model == "Doc2Vec":
            return self.distributions_doc2vec()
        else:
            print("The specified embedding model is not yet included in the package. We will look into it.")