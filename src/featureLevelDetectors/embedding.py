from typing import Callable, Dict, Optional, Union
import numpy as np
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from scipy import stats

from base import detectorParent
from baseModels import baseModels 
from sampling import samplingData

class embedding(samplingData, detectorParent):
    def __init__(self, *args, **kwargs):
        detectorParent.__init__(self, *args, **kwargs)
        samplingData.__init__(self, *args, **kwargs)
        """
        In this class, we turn the samples of text inputs into text embeddings, which we can then use
        to a) either construct distributions, or b) calculate drift on. There are many different kinds 
        of text embeddings and encodings. In this class, we cover 3 umbrella embeddings (discussed below)
        
        Returns
        ----------  
        a dictionary containing the embeddings as decided by the choice of embedding model and 
        drift detection test type
        """
    
    def sampleData(self):
        if self.sample_dict is None:
            return samplingData.samples(self)
        else:
            return self.sample_dict
   
    def embed_data(self):
        """
        Embeds text inherited from the sampling class.

        Returns
        ----------  
        a dictionary containing the embeddings as decided by the choice of embedding model and 
        drift detection test type
        """
        sample_dict = self.sampleData()
        data_ref = sample_dict[0]
        
        # need to look into what sort of data to inlude in tagged documents (for now just the first X text pieces)
        bases = baseModels(data = data_ref[:self.sample_size], sample_size = self.sample_size, 
                                SBERT_model = self.SBERT_model)
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
