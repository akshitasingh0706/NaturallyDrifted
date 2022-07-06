from typing import Callable, Dict, Optional, Union
import numpy as np
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer

from sampling import samplingData
from embedding import embedding

model_name = "bert-base-uncased"
class baseModels:
    def __init__(self, 
                data, sample_size, 
                 model_name: Optional[Union[str, None]]):
        self.data = data
        self.sample_size = sample_size
        self.model_name = model_name

    def doc2vec_base(self, 
                    # documents,
                    vector_size: Optional[int] = 100,
                    window: Optional[int] = 2,
                    min_count: Optional[int] = 1,
                    workers: Optional[int] = 4):
        '''
        sample_ref: dataset to get samples for "Tagged Documents"
        Doc2Vec params:
            vector_size
            window
            min_count
            workers
        '''
        documents = [TaggedDocument(np.random.choice(self.data, self.sample_size), [i]) for i, doc in enumerate(common_texts)]
        model = Doc2Vec(documents, vector_size = 100, window = 2, min_count = 1, workers = 4)
        return model

    def sbert_base(self):
        model = SentenceTransformer(self.model_name)
        return model