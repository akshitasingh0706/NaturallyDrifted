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
                embedding_model: Union["Doc2Vec", "SBERT", "USE"] = "Doc2Vec",
                SBERT_model: Optional[Union[str, None]] = None,
                iterations: int = 500,
                transformation: Optional[Union["PCA", "SVD", "UMAP", "UAE", None]] = None
                 ):
        """
        In this class, we construct distributions out of the embeddings we got from the "embedding" class. 
        This is an optional class and is only required if we are running a distribution dependent test such 
        as KLD or JSD. 
        
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
        
        transformation: str (optional)
            Embeddings render multiple multi-dimensional vector spaces. For instance, USE results in 512 
            dimensions, and 'bert-base-uncased' results in 768 dimensions. For feature levels tests such 
            as KLD or JSD, such a large dimension might not be feasible to analyse, and thus we can reduce 
            the dimensionality by selecting the most important components using methods such as PCA and SVD.

        iterations: int
            We can run through multiple iterations of the embeddings to make our drift detection test more
            robust. For instance, if we only detect a drift on 1 out of 10 itertions, then we might be 
            better off not flagging a drift at all.  

        Returns
        ----------  
        A dictionary containing the distributions as decided by the choice of embedding model and 
        drift detection test type
        """
        self.data_ref = data_ref
        self.data_h0  = data_h0
        self.data_h1  = data_h1
        self.sample_size = sample_size
        self.windows = windows
        self.drift_type = drift_type
        self.embedding_model = embedding_model
        self.SBERT_model= SBERT_model
        self.iterations = iterations
        self.transformation = transformation
        self.test = test

        self.bandwidth = .05
        self.kernel = 'gaussian'

    def kde(self): # ex. (bandwidth = .05, kernel = 'gaussian')
        return KernelDensity(bandwidth = self.bandwidth, kernel = self.kernel)

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
            embs = embedding(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, test = self.test,
                            sample_size = self.sample_size, windows = self.windows, drift_type = self.drift_type, 
                            embedding_model = self.embedding_model, SBERT_model = self.SBERT_model, 
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

    def distributions_seneconders(self):
        """
        Constructs distributions for SBERT or USE embeddings 

        Returns
        ----------  
        a dictionary containing the distributions as decided by the choice of embedding model and 
        drift dete
        """
        embs = embedding(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1, test = self.test, 
                        sample_size = self.sample_size, windows = self.windows, drift_type = self.drift_type, 
                        embedding_model = self.embedding_model, SBERT_model = self.SBERT_model, transformation = self.transformation)
        final_dict = embs.final_embeddings()  
        kde = KernelDensity(bandwidth = .05, kernel = 'gaussian')
        distributions_across_iters = {}
        for it in range(self.iterations):
            # if self.emb_dict is None:
            embs = embedding(data_ref = self.data_ref, data_h0 = self.data_h0, data_h1 = self.data_h1,
                             sample_size = self.sample_size, windows = self.windows, drift_type = self.drift_type, 
                            SBERT_model = self.SBERT_model, transformation = self.transformation, test = self.test,
                            embedding_model = self.embedding_model, iterations = None)
            emb_dict = embs.final_embeddings()
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