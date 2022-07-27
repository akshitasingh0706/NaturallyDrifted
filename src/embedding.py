from typing import Callable, Dict, Optional, Union
import numpy as np
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from scipy import stats

from baseModels import baseModels 
from sampling import samplingData

class embedding:
    def __init__(self, 
                data_ref: Optional[Union[np.ndarray, list, None]],
                data_h0: Optional[Union[np.ndarray, list, None]],
                data_h1: Union[np.ndarray, list], # "data" in sample_data_gradual
                sample_dict: Optional[Dict] = None,
                test: Union["KS", "KL", "JS"] = 'JS',
                sample_size: int = 500, 
                windows: Optional[int] = 20,
                drift_type: Optional[Union["Sudden", "Gradual"]] = "Sudden",
                embedding_model: Union["Doc2Vec", "SBERT", "USE"] = "Doc2Vec",
                SBERT_model: Optional[Union[str, None]] = None, 
                transformation: Union["PCA", "SVD", "UMAP", None] = None,
                iterations: Optional[int] = None):
        """
        In this class, we turn the samples of text inputs into text embeddings, which we can then use
        to a) either construct distributions, or b) calculate drift on. There are many different kinds 
        of text embeddings and encodings. In this class, we cover 3 umbrella embeddings (discussed below)
        
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
            as KLD or JSD, the 

        iterations: int
            We can run through multiple iterations of the embeddings to make our drift detection test more
            robust. For instance, if we only detect a drift on 1 out of 10 itertions, then we might be 
            better off not flagging a drift at all.  

        Returns
        ----------  
        a dictionary containing the embeddings as decided by the choice of embedding model and 
        drift detection test type
        """

        self.data_ref = data_ref
        self.data_h0  = data_h0
        self.data_h1  = data_h1
        self.sample_dict = sample_dict
        self.test = test
        self.sample_size = sample_size
        self.windows = windows
        self.drift_type = drift_type
        self.embedding_model = embedding_model
        self.SBERT_model = SBERT_model
        self.transformation = transformation
        self.iterations = iterations # relevant for KS which never gets to distributions

    def embed_data(self):
        """
        Embeds text inherited from the sampling class.

        Returns
        ----------  
        a dictionary containing the embeddings as decided by the choice of embedding model and 
        drift detection test type
        """
        if self.sample_dict is None:
            sample = samplingData(self.data_ref, self.data_h0, self.data_h1, 
                                self.drift_type, self.sample_size, self.windows)
            sample_dict = sample.samples()
            data_ref = sample_dict[0]
        data_ref = self.data_ref
        
        # need to look into what sort of data to inlude in tagged documents (for now just the first X text pieces)
        bases = baseModels(data = data_ref[:self.sample_size], sample_size = self.sample_size, SBERT_model = self.SBERT_model)
        emb_dict = {}

        if self.embedding_model == "Doc2Vec":
            model = bases.doc2vec_base()
            for i in range(len(sample_dict)):
                emb_dict[i] = model.infer_vector(sample_dict[i])
        elif self.embedding_model == "SBERT":
            model = bases.sbert_base()
            for i in range(len(sample_dict)):
                emb_dict[i] = model.encode(sample_dict[i])
        elif self.embedding_model == "USE":
            model = bases.use_base()
            for i in range(len(sample_dict)):
                emb_dict[i] = np.array(model(sample_dict[i]))
        else:
            print("The model is not defined")
        return emb_dict 
                  
    # only required for KS Test (which does not get to Distributions which is where we actually do iterations)
    def embed_data_iters(self):
        emb_dict = {}
        for i in range(self.iterations):
            temp_dict = self.embed_data()
            emb_dict[i] = temp_dict
        return emb_dict

    # constructed with SBERT in mind
    def dim_reduction(self,
                    emb_dict: Optional[dict] = None,
                    components: Optional[int] = 25,
                    n_iters: Optional[int] = 7):
        """
        Embeds text inherited from the sampling class.

        Args
        ---------- 
        emb_dict: dictionary
        Dictionary of embeddings as returned by the embed_data method
        component: int (optional)
        The number top components we want from PCA or SVD
        n_iters: int

        Returns
        ----------  
        a dictionary containing the embeddings as decided by the choice of embedding model and 
        drift detection test type
        """

        emb_dict = self.embed_data()
        if self.transformation == "PCA":
            model = PCA(n_components=components)
        elif self.transformation == "SVD":
            model = TruncatedSVD(n_components=components, n_iter= n_iters, random_state=42)
        else: 
            print("The following dimension reudction technique is not yet supported")
        '''
        Doc2Vec is a little more complicated so we will skip dim-reduction with it for now
        '''
        # only looking at the first iteration for now
        final_dict = {}
        for window in range(len(emb_dict)):
            model.fit(emb_dict[window].T)
            final_data = np.asarray(model.components_).T
            final_dict[window] = final_data
        return final_dict
      
    def final_embeddings(self):
        """
        Returns
        ----------  
        a dictionary containing the embeddings as decided by the choice of embedding model and 
        drift detection test type
        """
        if self.embedding_model == "Doc2Vec":
            if self.test == "KS":
                return self.embed_data_iters()
            elif self.test in ["KL", "JS"]:
                return self.embed_data()
            else:
                print("This test is not included in the package as yet")
        elif self.embedding_model in ["SBERT", "USE"]:
            if self.transformation is None:
                return self.embed_data()
            else:
                return self.dim_reduction()
        else:
            print("This embedding is not part of the package as yet.")
