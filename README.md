# Unsupervised Sentiment Analysis - Trustpage Challenge

## Developing unsupervised learning for movie reviews from IMDB
### Clustering problem
The dataset includes 50,000 reviews with positive/negative sentiment. The task is to develop an unsupervised learning model that can separate each review by learning the language structure and words were to distinguish positive reviews from negative ones. Thus, we have a binary clustering problem with two possible outcomes of positive and negative.
### Inspecting the dataset
I started the process by inspecting the data. Clearly, the reviews show the language used is the language on social media and not formal English. For example, the language could be sarcastic and descriptive. Some reviews, despite having words with positive meaning, overall state a negative statement. Also, there are HTML tags within some text, special characters, abbreviated forms of the words, etc. 


|   | review                                            | sentiment |   |   |
|---|---------------------------------------------------|-----------|---|---|
| 0 | One of the other reviewers has mentioned that ... | positive  |   |   |
| 1 | A wonderful little production. <br /><br />The... | positive  |   |   |
| 2 | I thought this was a wonderful way to spend ti... | positive  |   |   |
| 3 | Basically there's a family where a little boy ... | negative  |   |   |
| 4 | Petter Mattei's "Love in the Time of Money" is... | positive  |   |   |

### What ML algorithms to use for the clustering problem
Understanding the problem impact how to handle the data, evaluate the performance, and what ML algorithm to choose, I addressing this challenge. I decided to use K-Means clustering, clean and preprocess the text data accordingly, and evaluate the performance using the sentiments provided. 
Clustering is a common unsupervised machine learning in which similar data points are grouped. The created groups or clustered are homogenous subgroup data with similar characteristics. Unlike classification, in clustering, the label for data is not provided at the time of training. K-Means Clustering is a method that subdivides a dataset into K many different clustering. The algorithm analyzes the data to find similarities and assign each point to an appropriate cluster. Then, each cluster will be used to label the data into different clusters. In this algorithm, K Means constantly trying to find a centroid with data points based on the distance to data points. Thus, each assigned data point in each cluster is closer to that centroid compared to other centroids. 

<img src="https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2018/11/02/k-means-sagemaker-1.gif" alt="K-Means Clustering" style="height: 400px; width:400px;"/>

To apply K-Means Clustering:
* Select the appropriate number of clusters or K – based on the situation or using different modeling such as the elbow method to find the optimum number of clusters
    * for this clustering problem, I decided to use K = 2 as the goal is to cluster the review into 2 clusters of positive and negative clusters
* Initialize random centroids and calculated each data point distance to the centroids
* Assign new centroids by averaging over the data in each cluster, and then repeat this process till some tolerance such as number of iterations or convergence criteria is met

### Text Preprocessing
The text dataset contains only review text and must be transferred using NLP to extract the basic features and build the initial modeling. The dataset includes 50,000 rows in the data. 
Preprocessing and cleaning the data started with 
    * removing HTML tags
    * removing special characters
    * expanding the abbreviate and short forms
    * …
I used the bag of word technique to extract features from the movie reviews technique. Bag of words analyze the entire review dataset, builds a dictionary of all words, and creates a list of numbers for each word, and counts how many times each word appears in the dataset through a process called vectorization.
I used CountVectorizer from sklearn.feature_extraction.text to vectorize the data. it records the raw frequency of token occurrence and creates a sparse matrix with 50,000 rows and columns equal to the number of words in the whole text. 



### Basic Modeling – Sparse dataset
I created a simple working model using KMenas. I vectorize the dataset using CountVectorizer from sklearn.feature_extraction.text with:
* Stopwords: English – a built-in stop word list for English. Stop words like “and”, “the”, … are assumed to have no informative feature in representing the text. Thus, they need to b removed. There are few issues in stopwords used in vectorizer, such as words like “they’ve”. The vectorizer, by default, separates them but might include ‘they’ as a stop word but not ‘ve’. To address this issue, in the preprocessing, I first extended all the short forms like “they’ve” to “they have” and then split them. Therefore, Vectorizer should be able to catch those as stop words. 
* max_features:  build a vocabulary that only considers the top max_features ordered by term frequency across the corpus. For the primary run, I used 20000
The resulting array is sparse, as seen below. 

|   | killers | killian | killing | killings | killjoy | kills | kilmer | kilter | kim | kimberly |
|---|---------|---------|---------|----------|---------|-------|--------|--------|-----|----------|
| 0 | 0       | 0       | 0       | 0        | 0       | 0     | 0      | 0      | 0   | 0        |
| 1 | 0       | 0       | 0       | 0        | 0       | 0     | 0      | 0      | 0   | 0        |
| 2 | 0       | 0       | 1       | 0        | 0       | 0     | 0      | 0      | 0   | 0        |
| 3 | 0       | 0       | 0       | 0        | 0       | 0     | 0      | 0      | 0   | 0        |
| 4 | 0       | 0       | 0       | 0        | 0       | 0     | 0      | 0      | 0   | 0        |
| 5 | 0       | 0       | 0       | 0        | 0       | 0     | 0      | 0      | 0   | 0        |
| 6 | 0       | 0       | 0       | 0        | 0       | 0     | 0      | 0      | 0   | 0        |
| 7 | 0       | 0       | 0       | 0        | 0       | 0     | 0      | 0      | 0   | 0        |
| 8 | 0       | 0       | 0       | 0        | 0       | 0     | 0      | 0      | 0   | 0        |
| 9 | 0       | 0       | 0       | 0        | 0       | 0     | 0      | 0      | 0   | 0        |

Initially, I applied TruncatedSVD on the resulting sparse matrix to reduce the dimensionality. It is similar to PCA. However, the PCA centers the data before computing the singular value decomposition. Thus I used TruncatedSVD. Then, I applied the K-Means clustering on the resulting array from TruncatedSVD. 
* TruncatedSVD is a technique to reduce the dimensionality of the sparse dataset by finding sparse components that can optimally reconstruct the data. I used the following parameters for PCA analysis:
 * n_componentsint = 100, Desired dimensionality of output data. As recommended for LSA (latent semantic analysis), I used 100.
After performing TruncatedSVD, the transformed vec was fitted by K-Means, and then prediction was made using K.means.predict
~~~ Python
fitted = kmeans.fit(vec_tranformed)
prediction = kmeans.predict(vec_tranformed)
~~~

I used these values for K-Means:
* n_clusters = 2, the number of clusters and centroids to generate. Since the objective is to cluster the data into two subgroups, I used 2
* max_iter =1000, Maximum number of iterations of the k-means algorithm for a single run.
* n_initint, default=10, Number of times the k-means algorithm will be run with different centroid seeds. I used the default value

I run the model for two cases with TruncatedSVD and without. The results are shown below:
<font color=blue>
    
|                      | F1   | Accuracy  |
|----------------------|------|-----------|
| With TruncatedSVD    | 0.34 | 0.45      |
| Without TruncatedSVD | 0.26 | 0.51      |

</font>

The running time and performance slightly increased without using it. To improve the model, then I used the TfidVectorizer from sklearn.feature_extraction.text. Convert a collection of raw documents to a matrix of TF-IDF features. In practice, it is the same as CountVectorizer followed by TfidfTransformer. 
TfidfTransformer is a count matrix with normalized tf or tf-idf representation of its elements which are words. Tf means term-frequency, and tf-idf is used as term-frequency times inverse document-frequency. Unlike the CountVectorizer, which records the raw frequency of token occurrence, tf-idf also scales down the impact of tokens that frequently occur in a given corpus. So basically, using tf-idf tokenizer, the effect of frequently occurring words that are often less informative than features that happen in a small fraction of the training corpus will be minimized. 
I initialized the parameters of K-Means and TfidVectorizer as before. The results are shown in the table below:

<font color=blue>
    
|                      | F1   | Accuracy  |
|----------------------|------|-----------|
| With TruncatedSVD    | 0.65 | 0.57 |
| Without TruncatedSVD | 0.65 | 0.57      |


</font>


### Tuning the parameters for modeling 
In order to improve the performance of the model, a combination of three different values for each parameter was tested, a total number of 27 models for each case. Since the truncated svd partially reduced the accuracy for the first case and did not change the second case, I didn’t include it in this session. However, in unsupervised learning tuning, the parameters can be considered as performing supervised tuning. 
* max_fatures is the maximum number of world columns – 10000, 50000, None
* min_df minimum number of times a word must appear in the text – 1, 2, 3
* max_iter maximum number of iteration for KMenas – 1000, 2000, 3000
After running 27 different cases for both scenarios, I observed:
   * tuning practically had no impact on CountVectorizer – The choice of different parameters changes the accuracy slightly. 
   * tuning practically had little impact on TfidVectorizer with increasing the accuracy to 57%

### Selecting TfidVectorizer with max_feature=None, min_df=1, max_iter=3000

Using the results from the last session, I run TfidVectorizer case to create visualization. The final performance was
<font color=blue>
    
|                      | F1   | Accuracy  |
|----------------------|------|-----------|
| With TruncatedSVD    | 0.65 | 0.57 |

</font>







