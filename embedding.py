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
                test: Union["KS", "KL", "JS"],
                sample_size: int = 500, 
                windows: Optional[int] = 20,
                drift_type: Optional[Union["Sudden", "Gradual"]] = "Sudden",
                embedding_model: Union["Doc2Vec", "SBERT"] = "Doc2Vec",
                model_name: Optional[Union[str, None]] = None, 
                transformation: Union["PCA", "SVD", "UMAP", None] = None,
                iterations: Optional[int] = None):
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
        self.iterations = iterations # relevant for KS which never gets to distributions

    def embed_data(self):
        sample = samplingData(self.data_ref, self.data_h0, self.data_h1, 
                               self.drift_type, self.sample_size, self.windows)
        sample_dict = sample.samples()
        # need to look into what sort of data to inlude in tagged documents (for now just the first X text pieces)
        bases = baseModels(data = self.data_ref[:self.sample_size], sample_size = self.sample_size, model_name = self.model_name)
        emb_dict = {}

        if self.embedding_model == "Doc2Vec":
            model = bases.doc2vec_base()
            for i in range(len(sample_dict)):
                emb_dict[i] = model.infer_vector(sample_dict[i])
        elif self.embedding_model == "SBERT":
            model = bases.sbert_base()
            for i in range(len(sample_dict)):
                emb_dict[i] = model.encode(sample_dict[i])
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
      if self.embedding_model == "Doc2Vec":
          if self.test == "KS":
              return self.embed_data_iters()
          elif self.test in ["KL", "JS"]:
              return self.embed_data()
          else:
              print("This test is not included in the package as yet")
      if self.embedding_model == "SBERT":
          if self.transformation is None:
              return self.embed_data()
          else:
              return self.dim_reduction()