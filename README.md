NaturallyDrifted is drift detection library that focusses on drift detection in text data. 

# Table of Contents
1. [Installation and Usage](#installation-and-usage)
- [Google Colab Notebooks](#google-colab)
- [Azure/Databricks](#azure)
- [Local Environment (Jupyter Notebooks)](#jupyter)
- [Note: How to select data for analyses](#select-data)
- [Note: Review of Other function parameters](#other-func-params)
2. [Drift Detector Fundamentals](#drift-detector-fundamentals)
- [Types of Drifts (Based on Data, and Based on Time](#drift-types)
- [Covariate Drifts - Drift Detection Tests](#covariate-drift-tests)
- [Prior Drifts - Drift Detection Tests](#prior-drift-tests)
- [Concept Drifts - Drift Detection Tests](#concept-drift-tests)
- [Embedding Models - Text Data](#embedding-models)

<a name="installation-and-usage"/>

# Installation and Usage

<a name="google-colab"/>

## Google Colab

### Importing Code directly into Google Drive 
#### Step 1: Loading packaged and setting up the environment
- Import the Drift Detection folder onto Google Drive. You can clone the repo or download the folder from the following [Github repo](https://github.com/akshitasingh0706/DriftDetection). If you are new to Github, [this resource](https://docs.github.com/en/get-started) might come in useful. 
- Launch a new Google Colab notebook. If you are new to Colab, [this tutorial](https://colab.research.google.com/?utm_source=scs-index) might come in useful. 
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
#### Step 2: Loading and processing the Data
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
#### Step 3: Drift Detection
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

<a name="azure"/>

## Azure/Databricks
[To be completed]

<a name="jupyter"/>

## Local Environment (Jupyter Notebooks)
#### Step 1: Loading packaged and setting up the environment
- Set up your virtual environment and cd into it. For users with Macs with M1 chips, setting up the environment might be a little bit more involved. There are some online resources that help can help you set up Tensorflow/Transformers on Mac 1 such as this [article](https://towardsdatascience.com/hugging-face-transformers-on-apple-m1-26f0705874d7)
<pre>
filepath = [path to venv]
cd filepath
conda activate [name of venv]
</pre>
- Load the Drift Detection folder/clonded repo into the same folder as your virtual environment. 
- Launch a jupyter notebook
<pre>
jupyter notebook
</pre>
- cd into the Drift Detection folder
- Load the relevant packages
<pre>
pip install -r requirements.txt
</pre>
- Load additional python functions and classes
#### Steps 2 and 3 are the same as the ones in Google Colab section

<a name="select-data"/>

## A note on selecting data for analyses
To detect drifts, we need to look into the "reference data" as well as the comparison data. A convenient (but not the only) way to divide our data for our analyses is as follows:
#### data_ref
(type: np.ndarray, list)
- This is the dataset on which is used as the reference/baseline when detecting drifts. For instance, if our test of choice is KL Divergence, then we will declare a possible drift based on whether any other data is close in distribution to data_ref. 
- Generally, the goal is to have all future datasets be as close (in embeddings, distributions)
to data_ref, which is how we conclude that there is no drift in the dataset.  
- data_ref is typically sampled from the "training data". During real world application, 
this is the data on which the test will be modeled on because this would generally be 
the only data the user would have access to at that point of time. 

#### data_h0
(type: np.ndarray, list (optional))
- This is generally the same dataset as data_ref (or a stream that comes soon after). We use the lack of drift in data_h0 (with data_ref as our reference) as the necessary condition to decide the robustness of the drift detection method.
- If the method ends up detecting a drift in data_h0 itself, we know it is most likely not doing a good job. This is because both data_ref and data_h0 are expected to be coming from the same source and hence should result in similar embeddings and distributions. If the user is confident in the efficacy of their drift detection method, then it would be worthwhile to consider change the size of data_ref and data_h0 and then re-evaluate detector performance, before proceeding to data_h1. 

#### data_h1
(type: np.ndarray, list)
- This is the primary dataset on which we can expect to possibly detect a drift. In the real world, this would usually be the dataset we get post model deployment. To test detectors, a convenient (but not necessarily the best) practice is to take the test data and use that as our proxy for the deployed dataset. 
- Multiple research papers and libraries tend to also use "perturbed" data for their choice of data_h1. Perturbations can include corruptions in images (vision data) or introduction of unneccessary words and phrases (text data). This is generally the first step in testing the efficacy of a drift detection method. Once again, if the detectors fails to detect a drift on manually perturbed data, then its quite likely it will not be able to detect drifts in the real, deployed data as well. 
- Therefore, for our purposes, we have tried to minimize the use of artifically perturbed data
and instead rely on test data/data from far away time periods as our data_h1 source. 

<a name="other-func-params"/>

## Note: Decription of other relevant parameters
#### test: str
Specify the kind of drift detection test we want: "KS", "KL", "JS", "MMD", "LSDD" (discussed below).

#### sample_size: int
Decides the number of samples from each of the above 3 datasets that we would like to work with. For instance, if the entire training data is 100K sentences, we can use a sample_size = 500 to randomly sample 500 of those sentences. 

#### drift_type: str
Specify the drift type we are looking for, based on the time/frquency: "Sudden", "Gradual" (discussed below). 

#### windows: int (optional)
This parameter is only required for gradual/incremental drift detection. This decided the number of segments we would like to break the data into. 
For instance, if data_h1 has 100K data points, and if we wish to detect drifts gradually over time, a proxy approach would be to break the data in sets of 5K points and then randomly sample from each set separately. 

#### embedding_model: str
This parameter decides the kind of embedding the text goes through. The embeddings we consider thus far are: \\
a) SBERT: A Python framework for state-of-the-art sentence, text and image embeddings. \\ 
b) Universal Sentence Encoders: USE encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering, and other natural language tasks \\
c) Doc2Vec: a generalization of Word2Vec, which in turn is an algorithm that uses a 
neural network model to learn word associations from a large corpus of text

#### SBERT_model: str
This parameter is specific to the SBERT embedding models. If we choose to work with SBERT, we can specify the type of SBERT embedding out here. Ex. 'bert-base-uncased'

#### transformation:
Embeddings render multiple multi-dimensional vector spaces. For instance, USE results in 512 dimensions, and 'bert-base-uncased' results in 768 dimensions. For feature levels tests such  as KLD or JSD, such a large dimension might not be feasible to analyse, and thus we can reduce the dimensionality by selecting the most important components using methods such as PCA and SVD. 

#### iterations: int
We can run through multiple iterations of the embeddings to make our drift detection test more robust. For instance, if we only detect a drift on 1 out of 10 itertions, then we might be better off not flagging a drift at all.  

<a name="drift-detector-fundamentals"/>

# Drift Detection Fundamentals
![alt text](https://github.com/akshitasingh0706/DriftDetection/blob/trials/images/DDBasic.png?raw=true)

<a name="drift-types"/>

## Drift Types

### Based on Data (What drifted?)

![alt text](https://github.com/akshitasingh0706/DriftDetection/blob/trials/images/DriftTypes_Data.png?raw=true)

#### Covariate Drifts 
When the input data drifts: **P(X) != P<sub>ref</sub>(X)** even though **P(X|Y) != P<sub>ref</sub>(X|Y)**. Such drifts happen when the distribution of the input data, or some of the features, drifts. The drift can happen gradually, or right after deployment (discussed in the next section). For further reading, please refer to this [Seldon article](https://www.seldon.io/what-is-covariate-shift)

#### Prior Drifts 
When the output data drifts: **P(Y) != P<sub>ref</sub>(Y)** even though **P(X|Y) != P<sub>ref</sub>(X|Y)**. For instance, let's say trying to predict whether people with certain symptoms have COVID-19. Now, if we pick a training dataset before the pandemic, and test our model on a dataset during the pandemic, then our label distributions would be vastly different. The distribution of features of those people (ex. age, health parameters, location etc.) would be the exact same but the test data would just have a whole lot more labels that are COVID-19 positive heavy. 

#### Concept Drifts
When process generating *y* from *x* drifts:  **P(Y|X) != P<sub>ref</sub>(Y|X)**. Concept drift happens when the relationship between the input data (X) and outputs (Y) changes. 
\\
\\
For further reading, please refer to [Alibi Detect Documentation](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/background.html)

### Based on time/frequency (When drifted?)

![alt text](https://github.com/akshitasingh0706/DriftDetection/blob/trials/images/DriftTypes_Time.png?raw=true)

#### Sudden
These drifts generally happens quite instataneously right after deployment, likely because of a very immediate change in some external factors. Ex. a sudden drift in labels from "News" to "Sports" right after an election and right before the Olympics. 

#### Gradual
These drifts, as the name suggests, happen more gradually over time. These drifts might not be very obvious right away and depending on our thresholds, there is a possibility we might not catch them especially in the earlier time periods. One example could be the gradual drift in labels as we go from the peek of the pandemic all the way to it trough. Often such outbreaks reduce in intensity over time and hence the drift in labels might be more gradual. 

#### Incremental
Incremental drifts are similar to gradual drifts, with the additional consitency. This means that drift increases consistently over time, unlike the possible drops here and there in gradual drifts. 

#### Recurrent
Recurrent drifts are drifts wherein the model requires perpetual retraining. These drifts can be challening to work with as they can be hard to identify and will often require both our reference and comparison data to be updated after certain time intervals. For the previous 3 drifts, where the reference dataset stayed constant and we tested for Sudden or Gradual drifts on all the dataset that came after that. But for recurrent drifts, we cannot pick all data into eternity to test for drifts and will have to keep updating our information as we move in time.

<a name="covariate-drift-tests"/>

## Types of Drift Detection Tests - Covariate Drifts

### Feature Level

#### Kolmogorov–Smirnov (2-sample) test

![alt text](https://github.com/akshitasingh0706/DriftDetection/blob/trials/images/KS.png?raw=true)

A nonparametric test of the equality of continuous distributions. It quantifies a distance between the empirical distributions. For further reading, a possible resource you can refer to is this [article](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35g.htm)

#### Kullback–Leibler divergence

![alt text](https://github.com/akshitasingh0706/DriftDetection/blob/trials/images/KL_JS_MMD.jpeg?raw=true)

A distribution-dependent test that calculates the divergence between 2 distributions. [This resource](https://machinelearningmastery.com/divergence-between-probability-distributions/) gives a good overview on how we can implement KL divergence in Python. 

#### Jensen–Shannon divergence
A symmetric version of KL Divergence. It essent

###  Data Level

#### Maximum Mean Discrepency (MMD)

![alt text](https://github.com/akshitasingh0706/DriftDetection/blob/trials/images/MMD.png?raw=true)

A kernel based statistical test used to determine whether given two distribution are the same, first proposed in the paper ["A kernel two-sample test"](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf). MMD quantifies the distance between the mean embeddings (distribution map- ping into Reproducing Kernel Hilbert Space) of the distributions.This feature map reduces the distributions to simpler mathematical val- ues.

#### Least Squares Density Difference (LSDD)
LSDD is a method grounded in least squares, that estimates difference in distributions of a data pair without computing density distribution for each dataset indepen- dently. It was first proposed in the paper [Density Difference Estimation](https://direct.mit.edu/neco/article-abstract/25/10/2734/7921/Density-Difference-Estimation?redirectedFrom=fulltext)

#### Learned Kernel
[to be completed]

<a name="prior-drift-tests"/>

## Types of Drift Detection Tests - Prior Drifts
[to be completed]

<a name="concept-drift-tests"/>

## Types of Drift Detection Tests - Concept Drifts
[to be completed]

<a name="embedding-models"/>

## Types of Embedding Models

### Sentence Transformers (SBERT) 
SBERT is Python framework for state-of-the-art sentence, text and image embeddings. The initial work is described in the paper [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://www.sbert.net/). This framework can be used to compute sentence / text embeddings for more than 100 languages.

### Universal Sentence Encoders (USE)
USE encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering, and other natural language tasks. For further reading, refer to [Tensorflow-hub documentation](https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder)

### Document Embeddings (Doc2Vec/ Word2Vec, Glove)

#### Word2Vec
An algorithm that uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. For further reading, please refer to the paper [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

#### Doc2Vec
An NLP tool for representing documents as a vector and is a generalizing of the word2vec method. For further usage guidance, please refer to the [Gensim documentation](https://radimrehurek.com/gensim/models/doc2vec.html)
