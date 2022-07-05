from math import log2
import numpy as np
from scipy import stats
import pandas as pd
from typing import Callable, Dict, Optional, Union
import matplotlib.pyplot as plt

from embedding import embedding

from math import log2
class detectors:
    def __init__(self, 
                data_ref: Optional[Union[np.ndarray, list, None]],
                data_h0: Optional[Union[np.ndarray, list, None]],
                data_h1: Union[np.ndarray, list], # "data" in sample_data_gradual
                test: Union["KS", "KL", "JS"],
                sample_size: int = 500, 
                windows: Optional[int] = 20,
                drift_type: Optional[Union["Sudden", "Gradual"]] = "Sudden",
                embedding_model: Union["Doc2Vec", "SBERT"] = "Doc2Vec",
                model_name: Optional[Union[str, None]] = None, 
                transformation: Union["PCA", "SVD", "UMAP", None] = None,
                iters = 20):
        self.data_ref = data_ref
        self.data_h0  = data_h0
        self.data_h1  = data_h1
        self.test = test
        self.sample_size = sample_size
        self.windows = windows
        self.drift_type = drift_type
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.transformation = transformation
        self.iters = iters

    def kl_divergence(self, p, q):
        return sum(p[i] * log2(p[i]/q[i]) if p[i] > 0 and q[i] > 0 else 0 for i in range(len(p)))

    def js_divergence(self, p, q):
        return .5*(self.kl_divergence(p, q) + self.kl_divergence(q, p))

    '''
    function to calculate feature level tests:
        KL Divergence
        JS Divergence
        KS Test
    '''
    def ks_doc2vec(self):
        embs = embedding(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, test = self.test, 
                        sample_size = self.sample_size, windows = self.windows, drift_type = self.drift_type, 
                        embedding_model = self.embedding_model, model_name = self.model_name, 
                        transformation = self.transformation, emb_iters = self.iters)
        final_dict = embs.final_embeddings()  
        iterations = len(final_dict)
        windows = len(final_dict[0])
        dists = np.zeros((iterations, windows - 1)) # distances or p-values for (ref, every other window)
        pvals = np.zeros((iterations, windows - 1))
        for it in range(iterations):
            for ww in range(1, windows): 
                dists[it, ww -1] = stats.ks_2samp(final_dict[it][0], final_dict[it][ww])[0] 
                pvals[it, ww -1] = stats.ks_2samp(final_dict[it][0], final_dict[it][ww])[1] 

        # plotting results
        for window in range(pvals.shape[1]):
          plt.plot(pvals[:, window])
        plt.title("P-values for Doc2Vec + KS Test")
        plt.legend(["ww-"+ str(i) for i in range(pvals.shape[1])])
        return dists, pvals

    def ks_sbert(self):
        embs = embedding(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, test = self.test, 
                        sample_size = self.sample_size, windows = self.windows, drift_type = self.drift_type, 
                        model_name = self.model_name, transformation = self.transformation)
        final_dict = embs.final_embeddings()  
        windows = len(final_dict.keys())
        dimensions = final_dict[0].shape[1]
        dists = np.zeros((windows - 1, dimensions)) # distances or p-values for (ref, every other window)
        pvals = np.zeros((windows - 1, dimensions))
        for ww in range(1, windows): 
            for dim in range(dimensions):
                dists[ww -1, dim] = stats.ks_2samp(final_dict[0][:, dim], final_dict[ww][:, dim])[0] 
                pvals[ww -1, dim] = stats.ks_2samp(final_dict[0][:, dim], final_dict[ww][:, dim])[1] 

    def divergence_doc2vec(self):
        distrs = distributions(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, test = self.test, 
                        sample_size = self.sample_size, windows = self.windows, drift_type = self.drift_type, 
                        embedding_model = self.embedding_model, model_name = self.model_name, 
                        transformation = self.transformation, iterations = self.iters)
        final_dict = distrs.final_distributions() 

        iterations = len(final_dict)
        windows = len(final_dict[0])
        kld = np.zeros((iterations, windows - 1)) # distances or p-values for (ref, every other window)
        jsd = np.zeros((iterations, windows - 1))
        for it in range(iterations):
            for ww in range(1, windows): 
                kld[it, ww - 1] = self.kl_divergence(final_dict[it][0], final_dict[it][ww]) 
                jsd[it, ww - 1] = self.js_divergence(final_dict[it][0], final_dict[it][ww])

        # plotting KLD results
        if self.test == "KL":
            for window in range(kld.shape[1]):
                plt.plot(kld[:, window])
            plt.title("Distances for Doc2Vec + KL Divergence")
            plt.legend(["ww-"+ str(i) for i in range(kld.shape[1])])
            plt.show()

        # plotting JSD results
        if self.test == "JS":
            for window in range(jsd.shape[1]):
                plt.plot(jsd[:, window])
            plt.title("Distances for Doc2Vec + JS Divergence")
            plt.legend(["ww-"+ str(i) for i in range(jsd.shape[1])])
            plt.show()
        return kld, jsd

    def divergence_sbert(self):
        distrs = distributions(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, test = self.test,
                        sample_size = self.sample_size, windows = self.windows, drift_type = self.drift_type, 
                        model_name = self.model_name, embedding_model = self.embedding_model,
                        transformation = self.transformation, iterations = self.iters)
        final_dict = distrs.final_distributions() 
        iterations = len(final_dict.keys())
        windows = len(final_dict[0].keys())
        dimensions = final_dict[0][0][0].shape[0]
        kld = np.zeros((windows - 1, dimensions)) # distances or p-values for (ref, every other window)
        jsd = np.zeros((windows - 1, dimensions))

        # for it in range(iterations:
        for ww in range(1, windows): 
            for dim in range(dimensions):
                kld[ww -1, dim] = self.kl_divergence(final_dict[0][0][:, dim], final_dict[0][ww][:, dim]) 
                jsd[ww -1, dim] = self.js_divergence(final_dict[0][0][:, dim], final_dict[0][ww][:, dim])

        # plotting KLD results
        if self.test == "KL":
            for window in range(kld.shape[0]):
                plt.plot(kld[window, :30])
            plt.title("Distances for SBERT + KL Divergence")
            plt.legend(["ww-"+ str(i) for i in range(kld.shape[1])])
            plt.show()

        # plotting JSD results
        if self.test == "JS":
            for window in range(jsd.shape[0]):
                plt.plot(jsd[window, :30])
            plt.title("Distances for SBERT + JS Divergence")
            plt.legend(["ww-"+ str(i) for i in range(jsd.shape[1])])
            plt.show()
        return kld, jsd

    def detector(self):
        if self.test == "KS":
            if self.embedding_model == "Doc2Vec":
                return self.ks_doc2vec()
            if self.embedding_model == "SBERT":
                return self.divergence_sbert()

        elif self.test == "KL":
            if self.embedding_model == "Doc2Vec":
                return self.divergence_doc2vec()[0]
            if self.embedding_model == "SBERT":
                return self.divergence_sbert()[0]

        elif self.test == "JS":
            if self.embedding_model == "Doc2Vec":
                return self.divergence_doc2vec()[1]
            if self.embedding_model == "SBERT":
                return self.divergence_sbert()[1]
      
        else:
            print("The specified test or embedding model is not part of the package yet")
