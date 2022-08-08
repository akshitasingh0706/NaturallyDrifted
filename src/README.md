This folder contains all the source code required to run the detector function. 

### sampling.py
This file attempts to make data sampling for drift detection more efficient. Our data structure would be different based on whether we are interested in Sudden or Gradual drifts. This class helps in adding some level of convenience in how we can structure of our data, based on the kind of drifts we are interested in. The final output is a dictionary where each key represents a "data window". For Sudden Drifts for instance, we default to 3 windows (3 keys), representing data_ref, data_h0, and data_h1 (discussed in the main README.md)

### baseModels.py
This class sets the stage for the embedding models we choose to work with later. It returns a model framework based on the embedding model we are interested in working with. Currently, we have functionality for Doc2Vec, some Sentence Transformers and Universal Sentence Encoders. 

### embedding.py
In this class, we turn the samples of text inputs into text embeddings, which we can then use
to a) either construct distributions, or b) calculate drift as is. There are many different kinds 
of text embeddings and encodings. In this class, we cover 3 umbrella embeddings (discussed in baseModels.py). The main README.md gives a brief overview of each of these embeddings. The final output is dictionary of embeddings for each window. Each dict is wrapped around a larger dict which iterates through embeddings for different random samples. 

### distributions.py
In this class, we construct distributions out of the embeddings we got from the embedding class. 
This is an optional class and is only required if we are running a distribution dependent test such 
as KLD or JSD. The final output is dictionary of distributions for each window. Each dict is wrapped around a larger dict which iterates through distributions for different random samples. 

### myDetectors.py
This class returns the final detection results based on the embeddings or distributions it
inherits. Currently, the tests covered in this class are Kolmogorov–Smirnov test, 
Kullback–Leibler divergence, and Jensen-Shannon Divergence. Each test will return a different 
output based on the kind of embedding model we choose to work with. 

### AlibiDetectors.py
In this class, we implement some drift detection tests as developed by the Github Repo Alibi Detect.
Alibi Detect is a large repository with many different kinds of tests for different datasets (tabular,
vision, text). Here, we try to conveniently integrate some of the tests (MMD, LSDD) specifically 
for the text setting. 

### fileImports.py
This is an optional python file, which we can use in case if we wish to import all the other .py files directly into some location. (Ex. if we upload the folder into Google Drive)