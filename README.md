NaturallyDrifted is drift detection library that focusses on drift detection in text data. 

# Table of Contents
1. [Installation and Usage](#installation-and-usage)
- Conda
2. [Drift Detector Fundamentals](#drift-detector-fundamentals)


<a name="installation-and-usage"/>

# Installation and Usage

## Conda

### Importing Code directly into Google Drive 
- Import the Drift Detection folder onto Google Drive (https://github.com/akshitasingh0706/DriftDetection)
- Launch a new Google Colab notebook 
- Connect to Google Drive using the following commands: 
<pre>
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
</pre>
- Specify the path to Drift Detection folder in Google Drive
<pre>
filepath = [path to file]
# Ex: filepath = /content/gdrive/MyDrive/DriftDetection
</pre>
- Load the relevant packages
<pre>
!pip install -r filepath/requirements.txt
</pre>
- Load additional python functions and classes
<pre>
import sys
filepath = str(filepath)
sys.path.insert(0, filepath) # very important
from fileImports import imports 
samplingData, baseModels, embedding, distributions, detectors, AlibiDetectors = imports.run()
</pre>
- Load and process data. One example with the IMDB dataset is as follows:
<pre>
import nlp
def load_dataset(dataset: str, split: str = 'test'):
    data = nlp.load_dataset(dataset)
    X, y = [], []
    for x in data[split]:
        X.append(x['text'])
        y.append(x['label'])
    X = np.array(X)
    y = np.array(y)
    return X, y
X, y = load_dataset('imdb', split='train')
print(X.shape, y.shape)
</pre>
- Split it into the different pieces that will act as the reference and deployment data
<pre>
X1 = X[:round(X.shape[0]*.4)] # data_ref, data_h0
X2 = X[round(X.shape[0]*.4):] # data_h1
</pre>
- Check for Drift Detection. An example for Doc2Vec and KS test is given below
<pre>
# define variables/parameters 
drift_type = "Sudden"
model_name = None
embedding_model = "Doc2Vec" 
sample_size = 500
test = "JS"

# initialize the detector class with the above parameters
detectors = detectors(data_ref = X1, data_h0 = X1, data_h1 = X2, test = test, drift_type = drift_type,
                sample_size = sample_size, embedding_model = embedding_model, model_name = model_name)
# run the code to get the detector results
result = detectors.run() # change the term to "run" or "execute"
</pre>
- 
<a name="drift-detector-fundamentals"/>

# Drift Detection Fundamentals
![alt text](https://github.com/akshitasingh0706/DriftDetection/blob/trials/images/DDBasic.png?raw=true)

## Drift Types

### Based on Data (What drifted?)

![alt text](https://github.com/akshitasingh0706/DriftDetection/blob/trials/images/DriftTypes_Data.png?raw=true)

#### Covariate Drifts 
When the input data drifts: **P(X) != P<sub>ref</sub>(X)** even though **P(X|Y) != P<sub>ref</sub>(X|Y)** 

#### Prior Drifts 
When the output data drifts: **P(Y) != P<sub>ref</sub>(Y)** even though **P(X|Y) != P<sub>ref</sub>(X|Y)** 

#### Concept Drifts
When process generating *y* from *x* drifts:  **P(Y|X) != P<sub>ref</sub>(Y|X)**
\\
\\
For further reading, please refer to [Alibi Detect Documentation] (https://docs.seldon.io/projects/alibi-detect/en/stable/cd/background.html)


### Based on time/frequency (When drifted?)

![alt text](https://github.com/akshitasingh0706/DriftDetection/blob/trials/images/DriftTypes_Time.png?raw=true)

#### Sudden
A drift that happens quite suddenly, likely because of a very immediate change in some external factors. Ex. a sudden drift in labels from "News" to "Sports" right after an election and right before the Olympics. 

#### Gradual
A drift that happens more gradually over time. These drifts might not be very obvious right away and depending on our thresholds, there is a possibility we might not catch them especially in the earlier time periods. 

#### Incremental

#### Recurrent

# Covariate Drifts

## Types of Drift Detection Tests

### Feature Level
![alt text](https://github.com/akshitasingh0706/DriftDetection/blob/trials/images/KS.png?raw=true)
![alt text](https://github.com/akshitasingh0706/DriftDetection/blob/trials/images/KL_JS_MMD.png?raw=true)

#### Kolmogorov–Smirnov (2-sample) test
A nonparametric test of the equality of continuous distributions. It quantifies a distance between the empirical distributions. 

#### Kullback–Leibler divergence
A distribution-dependent test that calculates the divergence between 2 distributions. 

#### Jensen–Shannon divergence
A symmetric version of KL Divergence

###  Data Level

#### Maximum Mean Discrepency (MMD)
![alt text](https://github.com/akshitasingh0706/DriftDetection/blob/trials/images/MMD.png?raw=true)
A kernel based statistical test used to determine whether given two distribution are the same, first proposed in the paper "A kernel two-sample test" (https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf). MMD quantifies the distance between the mean embeddings (distribution map- ping into Reproducing Kernel Hilbert Space) of the distributions.This feature map reduces the distributions to simpler mathematical val- ues.

#### Least Squares Density Difference (LSDD)
LSDD is a method grounded in least squares, that estimates difference in distributions of a data pair without computing density distribution for each dataset indepen- dently. It was first proposed in the paper Density Difference Estimation (https://direct.mit.edu/neco/article-abstract/25/10/2734/7921/Density-Difference-Estimation?redirectedFrom=fulltext)

#### Learned Kernel

# Label Drifts

# Concept Drifts

## Types of Embedding Models

### Sentence Transformers (SBERT) 
is a Python framework for state-of-the-art sentence, text and image embeddings. The initial work is described in the paper [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks] (https://www.sbert.net/). This framework can be used to compute sentence / text embeddings for more than 100 languages.

### Universal Sentence Encoders (USE)
USE encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering, and other natural language tasks. For further reading, refer to [Tensorflow-hub documentation] (https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder)

### Document Embeddings (Doc2Vec/ Word2Vec, Glove)

#### Word2Vec
An algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. For further reading, please refer to the paper [Efficient Estimation of Word Representations in Vector Space] (https://arxiv.org/abs/1301.3781)


#### Doc2Vec
An NLP tool for representing documents as a vector and is a generalizing of the word2vec method. For further usage guidance, please refer to the [Gensim documentation] (https://radimrehurek.com/gensim/models/doc2vec.html)

#### Glove

