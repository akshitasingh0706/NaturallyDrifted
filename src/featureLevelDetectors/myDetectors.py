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
    def __init__(self, *args, **kwargs):
        """
        This class returns the final detection results based on the embeddings or distributions it
        inherits. Currently, the tests covered in this class are Kolmogorov–Smirnov test, 
        Kullback–Leibler divergence, and Jensen-Shannon Divergence. Each test will return a different 
        output based on the kind of embedding model we choose to work with. 

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

        embedding_model: str
            This is the principle parameter of this class. It decided the kind of embedding the text 
            goes through. The embeddings we consider thus far are: 
            a) SBERT: A Python framework for state-of-the-art sentence, text and image embeddings. 
            b) Universal Sentence Encoders: USE encodes text into high dimensional vectors that can be 
            used for text classification, semantic similarity, clustering, and other natural language tasks
            c) Doc2Vec: a generalization of Word2Vec, which in turn is an algorithm that uses a 
            neural network model to learn word associations from a large corpus of text
        
        Returns
        ----------
        Drift detection related test statistics and any relevant plots
        """
        super(myDetectors, self).__init__(*args, **kwargs)

    def kl_divergence(self, p, q):
        """
        Calculated the Kullback–Leibler Divergence for the 2 distributions p and q

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
        Calculated the Jensen–Shannon Divergence for the 2 distributions p and q

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
        Calculated Kolmogorov–Smirnov test for Doc2Vec embeddings

        Returns
        ----------    
        The p-values and distances as given by the Kolmogorov–Smirnov test   
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
        Calculated the Kolmogorov–Smirnov test test for SBERT embeddings. 

        Returns
        ----------    
        The p-values and distances as given by the Kolmogorov–Smirnov test    
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
        Calculated Kullback–Leibler or Jensen–Shannon Divergence for Doc2Vec embeddings

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
        Calculated Kullback–Leibler or Jensen–Shannon Divergence for SBERT/USE embeddings

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
                return self.ks_sbert()

        elif self.test == "KL":
            if self.embedding_model == "Doc2Vec":
                return self.divergence_doc2vec()['KL_div']
            if self.embedding_model in ["SBERT", "USE"]:
                return self.divergence_seneconders()['KL_div']

        elif self.test == "JS":
            if self.embedding_model == "Doc2Vec":
                return self.divergence_doc2vec()['JS_div']
            if self.embedding_model in ["SBERT", "USE"]:
                return self.divergence_seneconders()['JS_div']
      
        else:
            print("The specified test or embedding model is not part of the package yet")