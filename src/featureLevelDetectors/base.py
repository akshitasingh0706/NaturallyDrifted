from typing import Dict, Optional, Union
import numpy as np
from transformers import AutoTokenizer

class detectorParent:
    def __init__(self,
                # data specifications
                data_ref: Optional[Union[np.ndarray, list, None]] = None,
                data_h0: Optional[Union[np.ndarray, list, None]] = None,
                data_h1: Optional[Union[np.ndarray, list]] = None, # "data" in sample_data_gradual
                sample_dict: Optional[Dict] = None,

                # detector specifications
                test: Union["MMD", "LSDD"] = "MMD",
                sample_size: int = 500, 
                windows: Optional[int] = 10,
                drift_type: Optional[Union["Sudden", "Gradual"]] = "Sudden",
                SBERT_model: str = 'bert-base-uncased',   
                embedding_model: Union["Doc2Vec", "SBERT", "USE"] = "Doc2Vec",
                transformation: Union["UMAP", "UAE", None] = None,
                pval_thresh: int = .05,
                dist_thresh: int = .0009,
                iterations: int = 5,

                # other functions
                plot: bool = True,
                 ):
        """
        Define base arguments/parameters required by all Text related Alibi detectors. 
        
        Args
        ----------
        data_ref : np.ndarray, list
            Dataset on which model is trained (ex: training dataset). We compare a drift with a
            reference to this distribution.

        data_h0 :  np.ndarray, list (optional)
            Generally, the same dataset as data_ref (or a stream that comes soon after).
            We use the lack of drift in data_h0 (with data_ref as our reference) as the necessary 
            condition to decide the robustness of the drift detection method. 

        data_h1: np.ndarray, list
            Principal dataset on which we might see a drift (ex. deployment data). It can be just one
            sample (for sudden drifts) or stream of samples (for gradual drifts)
        
        sample_dict: dict
            Dictionary with samples for reference and comparison data (or streams of comparison data)

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
        Nothing
        """
        self.data_ref = data_ref
        self.data_h0  = data_h0
        self.data_h1  = data_h1
        self.sample_dict = sample_dict
        self.test = test
        self.sample_size = sample_size
        self.windows = windows

        self.drift_type = drift_type
        self.SBERT_model = SBERT_model
        self.embedding_model = embedding_model
        self.transformation = transformation
        self.pval_thresh = pval_thresh
        self.dist_thresh = dist_thresh
        self.iterations = iterations
        self.plot = plot    


