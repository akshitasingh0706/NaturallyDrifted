from math import log2
import numpy as np
from scipy import stats
import pandas as pd
from typing import Callable, Dict, Optional, Union
import matplotlib.pyplot as plt

from base import detectorParent
from baseModels import baseModels 
from sampling import samplingData
from baseModels import baseModels
from embedding import embedding
from distributions import distributions

class myDetectors(distributions, embedding, samplingData, detectorParent):
    # def __init__(self, *args, **kwargs):
    #     detectorParent.__init__(self, *args, **kwargs)
    #     samplingData.__init__(self, *args, **kwargs)
    #     """
    #     This class returns the final detection results based on the embeddings or distributions it
    #     inherits. Currently, the tests covered in this class are Kolmogorov–Smirnov test, 
    #     Kullback–Leibler divergence, and Jensen-Shannon Divergence. Each test will return a different 
    #     output based on the kind of embedding model we choose to work with. 
        # """
    def kl_divergence(self, p, q):
        """
        Calculated the KL Divergence for the 2 distributions p and q

        Args
        ----------
        p: np.ndarray
        A numpy array containing the distributions of some data
        q: np.ndarray
        A numpy array containing the distributions of some data

        Returns
        ----------    
        The KL Divergence distance    
        """
        return sum(p[i] * log2(p[i]/q[i]) if p[i] > 0 and q[i] > 0 else 0 for i in range(len(p)))

    def js_divergence(self, p, q):
        """
        Calculated the JS Divergence for the 2 distributions p and q

        Args
        ----------
        p: np.ndarray
        A numpy array containing the distributions of some data
        q: np.ndarray
        A numpy array containing the distributions of some data

        Returns
        ----------    
        The JS Divergence distance    
        """
        return .5*(self.kl_divergence(p, q) + self.kl_divergence(q, p))

    def ks_doc2vec(self):
        """
        Calculated KS test for Doc2Vec embeddings

        Returns
        ----------    
        The p-values and distances as given by the KS test    
        """
        final_dict = embedding.final_embeddings(self)
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

        test_stats = {}
        test_stats["distances"] = dists
        test_stats["pvalues"] = pvals
        return test_stats

    def ks_sbert(self):
        """
        Calculated KS test for SBERT embeddings

        Returns
        ----------    
        The p-values and distances as given by the KS test    
        """
        final_dict = embedding.final_embeddings(self) 
        windows = len(final_dict.keys())
        dimensions = final_dict[0].shape[1]
        dists = np.zeros((windows - 1, dimensions)) # distances or p-values for (ref, every other window)
        pvals = np.zeros((windows - 1, dimensions))
        for ww in range(1, windows): 
            for dim in range(dimensions):
                dists[ww -1, dim] = stats.ks_2samp(final_dict[0][:, dim], final_dict[ww][:, dim])[0] 
                pvals[ww -1, dim] = stats.ks_2samp(final_dict[0][:, dim], final_dict[ww][:, dim])[1] 

        test_stats = {}
        test_stats["distances"] = dists
        test_stats["pvalues"] = pvals
        return test_stats

    def divergence_doc2vec(self):
        """
        Calculated KL or JS Divergence for Doc2Vec embeddings

        Returns
        ----------    
        The distances as given by the KL or JS Divergence    
        """
        final_dict = distributions.final_distributions(self) 
        iterations = len(final_dict)
        windows = len(final_dict[0])
        kld = np.zeros((iterations, windows - 1)) # distances or p-values for (ref, every other window)
        jsd = np.zeros((iterations, windows - 1))
        for it in range(iterations):
            for ww in range(1, windows): 
                kld[it, ww - 1] = self.kl_divergence(final_dict[it][0], final_dict[it][ww]) 
                jsd[it, ww - 1] = self.js_divergence(final_dict[it][0], final_dict[it][ww])

        if self.plot == 'Yes':
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

        test_stats = {}
        test_stats["KL_div"] = kld
        test_stats["JS_div"] = jsd
        return test_stats

    def divergence_seneconders(self):
        """
        Calculated KL or JS Divergence for SBERT/USE embeddings

        Returns
        ----------    
        The distances as given by the KL or JS Divergence    
        """
        final_dict = distributions.final_distributions(self) 
        windows = len(final_dict[0].keys())
        dimensions = final_dict[0][0][0].shape[0]
        kld = np.zeros((windows - 1, dimensions)) # distances or p-values for (ref, every other window)
        jsd = np.zeros((windows - 1, dimensions))

        for ww in range(1, windows): 
            for dim in range(dimensions):
                kld[ww -1, dim] = self.kl_divergence(final_dict[0][0][:, dim], final_dict[0][ww][:, dim]) 
                jsd[ww -1, dim] = self.js_divergence(final_dict[0][0][:, dim], final_dict[0][ww][:, dim])

        if self.plot == 'Yes':
            # plotting KLD results
            if self.test == "KL":
                for window in range(kld.shape[0]):
                    plt.plot(kld[window, :])
                plt.title("Distances for " + self.embedding_model + " and " +  self.test)
                plt.legend(["ww-"+ str(i) for i in range(kld.shape[1])])
                plt.show()

            # plotting JSD results
            if self.test == "JS":
                for window in range(jsd.shape[0]):
                    plt.plot(jsd[window, :])
                plt.title("Distances for " + self.embedding_model + " and " +  self.test)
                plt.legend(["ww-"+ str(i) for i in range(jsd.shape[1])])
                plt.show()
        
        test_stats = {}
        test_stats["KL_div"] = kld
        test_stats["JS_div"] = jsd
        return test_stats

    def run(self):
        """
        Calculates the drift detection metrics, as specified by the choice of embedding model and 
        drift detection test. 

        Returns
        ----------    
        Distances for KLD or JSD or P-values for KS (depending on choice of test)
        """
        if self.test == "KS":
            if self.embedding_model == "Doc2Vec":
                return self.ks_doc2vec()
            if self.embedding_model in ["SBERT", "USE"]:
                return self.divergence_seneconders()

        elif self.test == "KL":
            if self.embedding_model == "Doc2Vec":
                return self.divergence_doc2vec()[0]
            if self.embedding_model in ["SBERT", "USE"]:
                return self.divergence_seneconders()[0]

        elif self.test == "JS":
            if self.embedding_model == "Doc2Vec":
                return self.divergence_doc2vec()['JS_div']
            if self.embedding_model in ["SBERT", "USE"]:
                return self.divergence_seneconders()['JS_div']
      
        else:
            print("The specified test or embedding model is not part of the package yet")