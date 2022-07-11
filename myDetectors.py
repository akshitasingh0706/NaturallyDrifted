from math import log2
import numpy as np
from scipy import stats
import pandas as pd
from typing import Callable, Dict, Optional, Union
import matplotlib.pyplot as plt

from sampling import samplingData
from baseModels import baseModels
from embedding import embedding
from distributions import distributions

class run:
    def __init__(self, 
                data_ref: Optional[Union[np.ndarray, list, None]],
                data_h0: Optional[Union[np.ndarray, list, None]],
                data_h1: Union[np.ndarray, list], # "data" in sample_data_gradual
                test: Union["KS", "KL", "JS"],
                sample_size: int = 500, 
                windows: Optional[int] = 20,
                drift_type: Optional[Union["Sudden", "Gradual"]] = "Sudden",
                embedding_model: Union["Doc2Vec", "SBERT", "USE"] = "Doc2Vec",
                model_name: Optional[Union[str, None]] = None, 
                transformation: Union["PCA", "SVD", "UMAP", None] = None,
                iterations = 20):
        """
        In this class, we turn the samples of text inputs into text embeddings, which we can then use
        to a) either construct distributions, or b) calculate drift on. There are many different kinds 
        of text embeddings and encodings. In this class, we cover 3 umbrella embeddings (discussed below)
        
        Args
        ----------
        data_ref : np.ndarray, list
            This is the dataset on which is used as the reference/baseline when detecting drifts. 
            For instance, if our test of choice is KL Divergence, then we will declare a possible
            drift based on whether any other data is close in distribution to data_ref. 
            Generally, the goal is to have all future datasets be as close (in embeddings, distributions)
            to data_ref, which is how we conclude that there is no drift in the dataset.  
            
            data_ref is typically sampled from the "training data". During real world application, 
            this is the data on which the test will be modeled on because this would generally be 
            the only data the user would have access to at that point of time. 

        data_h0 :  np.ndarray, list (optional)
            This is generally the same dataset as data_ref (or a stream that comes soon after).
            We use the lack of drift in data_h0 (with data_ref as our reference) as the necessary 
            condition to decide the robustness of the drift detection method. If the method ends up 
            detecting a drift in data_h0 itself, we know it is most likely not doing a good job. 
            This is because both data_ref and data_h0 are expected to be coming from the same source 
            and hence should result in similar embeddings and distributions. 

            If the user is confident in the efficacy of their drift detection method, then it would be 
            worthwhile to consider change the size of data_ref and data_h0 and then re-evaluate detector
            performance, before proceeding to data_h1. 

        data_h1: np.ndarray, list
            This is the primary dataset on which we can expect to possibly detect a drift. In the real 
            world, this would usually be the dataset we get post model deployment. To test detectors, a
            convenient (but not necessarily the best) practice is to take the test data and use that as
            our proxy for the deployed dataset. 

            Multiple research papers and libraries tend to also use "perturbed" data for their choice of
            data_h1. Perturbations can include corruptions in images (vision data) or introduction of 
            unneccessary words and phrases (text data). This is generally the first step in testing the 
            efficacy of a drift detection method. Once again, if the detectors fails to detect a drift
            on manually perturbed data, then its quite likely it will not be able to detect drifts in 
            the real, deployed data as well. 

            Therefore, for our purposes, we have tried to minimize the use of artifically perturbed data
            and instead rely on test data/data from far away time periods as our data_h1 source. 

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
        
        transformation:
            Embeddings render multiple multi-dimensional vector spaces. For instance, USE results in 512
            dimensions, and 'bert-base-uncased' results in 768 dimensions. For feature levels tests such 
            as KLD or JSD, the 

        iterations: int
            We can run through multiple iterations of the embeddings to make our drift detection test more
            robust. For instance, if we only detect a drift on 1 out of 10 itertions, then we might be 
            better off not flagging a drift at all.  
        """

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
        self.iterations = iterations

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
        embs = embedding(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, test = self.test, 
                        sample_size = self.sample_size, windows = self.windows, drift_type = self.drift_type, 
                        embedding_model = self.embedding_model, model_name = self.model_name, 
                        transformation = self.transformation, iterations = self.iterations)
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
        """
        Calculated KS test for SBERT embeddings

        Returns
        ----------    
        The p-values and distances as given by the KS test    
        """
        embs = embedding(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, test = self.test, 
                        sample_size = self.sample_size, windows = self.windows, drift_type = self.drift_type, 
                        model_name = self.model_name, transformation = self.transformation, 
                        iterations = self.iterations, embedding_model = self.embedding_model)
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
        """
        Calculated KL or JS Divergence for Doc2Vec embeddings

        Returns
        ----------    
        The distances as given by the KL or JS Divergence    
        """
        distrs = distributions(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, test = self.test, 
                        sample_size = self.sample_size, windows = self.windows, drift_type = self.drift_type, 
                        embedding_model = self.embedding_model, model_name = self.model_name, 
                        transformation = self.transformation, iterations = self.iterations)
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

    def divergence_seneconders(self):
        """
        Calculated KL or JS Divergence for SBERT/USE embeddings

        Returns
        ----------    
        The distances as given by the KL or JS Divergence    
        """
        distrs = distributions(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, test = self.test,
                        sample_size = self.sample_size, windows = self.windows, drift_type = self.drift_type, 
                        model_name = self.model_name, embedding_model = self.embedding_model,
                        transformation = self.transformation, iterations = self.iterations)
        final_dict = distrs.final_distributions() 
        windows = len(final_dict[0].keys())
        dimensions = final_dict[0][0][0].shape[0]
        kld = np.zeros((windows - 1, dimensions)) # distances or p-values for (ref, every other window)
        jsd = np.zeros((windows - 1, dimensions))

        for ww in range(1, windows): 
            for dim in range(dimensions):
                kld[ww -1, dim] = self.kl_divergence(final_dict[0][0][:, dim], final_dict[0][ww][:, dim]) 
                jsd[ww -1, dim] = self.js_divergence(final_dict[0][0][:, dim], final_dict[0][ww][:, dim])

        # plotting KLD results
        if self.test == "KL":
            for window in range(kld.shape[0]):
                plt.plot(kld[window, :])
            plt.title("Distances for", self.embedding_model, "and",  self.test)
            plt.legend(["ww-"+ str(i) for i in range(kld.shape[1])])
            plt.show()

        # plotting JSD results
        if self.test == "JS":
            for window in range(jsd.shape[0]):
                plt.plot(jsd[window, :])
            plt.title("Distances for", self.embedding_model, "and",  self.test)
            plt.legend(["ww-"+ str(i) for i in range(jsd.shape[1])])
            plt.show()
        return kld, jsd
    
    # plot can also be a parameter in the def run... method
    def plot(self):
        pass

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
                return self.divergence_doc2vec()[1]
            if self.embedding_model in ["SBERT", "USE"]:
                return self.divergence_seneconders()[1]
      
        else:
            print("The specified test or embedding model is not part of the package yet")
