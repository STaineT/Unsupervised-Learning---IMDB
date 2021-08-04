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
    
| | F1   | Accuracy  |
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

### Selecting TfidVectorizer with max_feature=None, min_df=2, max_iter=1000
Using the results from the last session, I run TfidVectorizer case to create visualization. The final performance was
<font color=blue>
    
    
|                      | F1   | Accuracy  |
|----------------------|------|-----------|
| With TruncatedSVD    | 0.65 | 0.57 |


</font>







### Basic Modeling – Dense matrix
In order to improve the outcome of the modeling, I used the Word2Vec package to create a dense matrix instead of the sparse matrix discussed before. Word2vec is a two-layer neural network that processes the text by vectorizing similar to other packages in the previous section. However, it creates a dense matric rather than of dimension equal to the number of words in the text and vector size of the array representing each word. 
There are two methods within the Word2vec algorithm:
* Continuous bag of words model – which predicts middle word based on surrounding context words, which are few words after and before. 
* Continuous skip-gram words model, which predicts words within a specific range before and after the current word. Skip-gram is the method that is used here.
I applied the basic modeling with default values using Wor2Vec and then KMeans clustering again with two clusters. Some of the parameters which I used are
* vector_size – Dimensionality of the word vectors, I used 300 and later 500
* window: 4 – Maximum distance between the current and predicted word within a sentence.
* min_count: 3– Ignores all words with total frequency lower than this.
* workers – Use these many worker threads to train the model (=faster training with multicore machines).
* sg ({0, 1}:1 – Training algorithm: 1 for skip-gram; otherwise CBOW.
* Negative: 20 – If > 0, negative sampling will be used. The int for negative specifies how many “noise words” should be drawn
Preprocessing the text was as before, cleaning the text. However, to use the word2vec method, we have to create sentences, a list of words.
* clean the text
* tokenized each word and removed the stop words
* build and trained the word2vec model over tokenized sentences
* To create a 2d array for K-means clustering, I defined a function to average each sentence. Each word is represented in a 1d array of size equal to vector size defined in word2pack. Every sentence has a different number of words. Thus, the averaging function is taking all the word vectors for each sentence, and after adding them together, it divides them by the length of that sentence to create an average vector of length vector_size.  The resulting average array was fed into the K-means clustering
The performance of the model was not satisfactory, and the initial run was led to 52%.


### Tuning the parameters for modeling 

Several adjustments were made into the initial model, including:
* Increasing the vector_size to 500
* defining three different tokenizers:
 * defined bigram to capture two words like in the text and treat them as a single word
 * defined a custom function to create stemmed word and lemmatized word, and remove stop words 
 * defined a custom function to remove stop words and tokenized each sentence 
 * increased the training epoch
All adjustments did not increase the accuracy and the highest accuracy achieved was 54%

### Apply Vader Lexicon model 

Vader Lexicon is a semi-supervised learning model and was used only for comparison. Accuracy of 67% was achieved using Vader.

## Improve the performance by trying a new idea – Kmeans clustering for each word

After trying several approaches, I decided to apply the K-Means to every word within the data frame to specify its cluster. Then, calculate its L2 norm, and multiply by the cluster number (-1,1) to calculate a score. The calculated scores for words in each row are averaged and can be used to indicate the positivity or negativity of the review. Thus the steps I take are:

* created a data frame from word_vector object out of the word2vec model, data frame with 300 number of features, and 60,000 rows
* extracted the word and associated vector of length 300 for each word
* applied K-Means on each word vector and calculate the cluster for each word
* assigned 1 for cluster 1 and -1 for cluster 0
* calculated the L2 norm for each word and multiplied it with the cluster number to calculate a parameter called word_score



| words | vectors |                                           cluster | cluster_number | l2_distance | word_score |
|------:|--------:|--------------------------------------------------:|---------------:|------------:|-----------:|
| words |         |                                                   |                |             |            |
|  will |    will | [-0.20066422, 0.049886666, 0.17076576, 0.21375... |              1 |           1 |   0.150745 |
| movie |   movie | [-0.38537654, 0.79678607, 0.18432134, 0.218949... |              1 |           1 |   0.167779 |
|  film |    film | [-0.21427089, 0.41739598, -0.34052396, 0.05714... |              1 |           1 |   0.193551 |
|   one |     one | [-0.091151565, -0.03470321, -0.07899916, -0.21... |              1 |           1 |   0.224714 |
|  like |    like | [-0.4568476, 0.41476494, 0.12726973, -0.324014... |              1 |           1 |   0.196834 |
|  just |    just | [-0.27087077, 0.5324964, -0.0018608796, -0.307... |              1 |           1 |   0.193294 |
|  good |    good | [0.30337575, 0.3011408, 0.2350973, 0.07215704,... |              1 |           1 |   0.169522 |
|  time |    time | [0.11032709, -0.023637589, 0.15891556, 0.53954... |              1 |           1 |   0.175976 |
|  even |    even | [-0.026722243, 0.37848318, -0.22821146, -0.405... |              1 |           1 |   0.205808 |


* I defined a function that looked up the word score from the table above for every word in every sentence and then took a mean of all the word scores within that sentence. If the number calculated was greater than 0, that indicates a positive review, and if it was negative, that suggested a negative review. 
* I notice that the length of the array created in the words table (output of the word2vec) is heavily dependent on the word tokenizer algorithm. It could have as many as 95,000 rows in the multiple analysis I performed. Generally, throughout all different scenarios which I explored, words tokenizations and text preparation had significant output. Therefore, I tried not to incorporate the accuracy performance on choosing the model parameters to avoid supervised parameter tuning and instead use values that seem more appropriate. 
* the result was not promising, and I expected a better performance. Accuracy was slightly above 50%, as the algorithm did poorly on negative reviews. The problem with my model, which impacted the final results, puts a lot of emphasis on the word score in the context of the whole document. 

## Conclusion and ways to improve the model

Although the accuracy of the final idea is not significant compared to semi-supervised learning models, it can still be improved. One way to improve the modeling is to incorporate the word score in the scope of each sentence level as well. Tf-Idf can be used to calculate another score for each word in each sentence. This implementation will ensure that enough emphasis is given to the words that are unique in each sentence. Then, the combination of this score and the one from above can improve the modeling. 


---
---

## Slight modification to boost the accuracy
In order to boost the accuracy of the model presented and remove unrelated words in each review that might incorrectly impact the clustering algorithm, I explored the idea of incorporating the results from the dense dataset into the sparse dataset.
There are many words in the reviews related to the movies or characters rather then positive and negative sentiment. In order to exclude these words, I made a slight adjustment to the submission, as explained below.
### Customizing the stop_wrods
* In the Dense dataset, in the original submission, I had explored the idea of clustering with more than 2 clusters to exclude the unrelated words. However, since it did not impact the final accuracy, I did not include it in the final documentation. 
* For improving the model performance, I implemented the dense dataset but with K=3 this time. I created a data frame using the resulted vector from word2vec model.


|        |                                           vectors | cluster | l2_distance | word_score |
|-------:|--------------------------------------------------:|--------:|------------:|-----------:|
|  words |                                                   |         |             |            |
|        | [-0.19455506, 0.28348318, 0.20605221, 0.195558... |       0 |    0.241723 |  -0.241723 |
|  movie | [-0.14058958, 0.5709055, -0.22287044, 0.329827... |       1 |    0.172073 |   0.172073 |
|   film | [-0.091226876, 0.10236713, -0.44825375, 0.2665... |       1 |    0.243680 |   0.243680 |
|    one | [-0.08742523, -0.025102125, 0.31197816, 0.4312... |       1 |    0.215025 |   0.215025 |
|   like | [-0.36074874, 0.7584568, -0.19159229, -0.25797... |       0 |    0.197997 |  -0.197997 |
|   just | [-0.24510749, 0.38719353, -0.011783284, 0.0007... |       1 |    0.190104 |   0.190104 |
|   good | [0.19814792, 0.8872247, 0.17918931, 0.14578643... |       1 |    0.174220 |   0.174220 |
|   time | [-0.18041916, 0.3175915, 0.27195337, 0.1915056... |       1 |    0.177395 |   0.177395 |
|   even | [-0.04309079, 0.20007537, -0.34513903, -0.3730... |       1 |    0.194733 |   0.194733 |
|   will | [-0.20239432, 0.4686843, 0.40891212, 0.2796957... |       0 |    0.155112 |  -0.155112 |
|  story | [0.051561855, 0.11486601, 0.11613593, 0.005188... |       1 |    0.163962 |   0.163962 |
| really | [-0.15956417, 0.5350447, -0.18707435, -0.09774... |       1 |    0.186785 |   0.186785 |
|    see | [-0.33177364, 0.57119185, 0.115189485, 0.00997... |       1 |    0.186898 |   0.186898 |
|    can | [-0.045773186, 0.48658392, -0.1992637, -0.1158... |       1 |    0.188658 |   0.188658 |
|   well | [0.19789116, 0.46953967, -0.010101889, 0.27403... |       1 |    0.181949 |   0.181949 |


* then I explored the three word clusters: 

~~~ Python
words_trimmed['cluster'].value_counts()
0    31109
1     9923
2     9907
Name: cluster, dtype: int64

~~~


* Visual inspection of each cluster showed that k=1 has words that are more related to the movies and characters than those indicating the writer's feeling or perception of the movies. 
* After finishing clustering each word, I incorporated cluster 1 into the stop_words for the sparse dataset. In order to ensure the valid assumption for cluster 1, I repeated the process for K=0, K=2 as well. However, the results confirmed that k=1 is an appropriate choice for eliminating irrelevant words.

* I created customized stop words by adding the words in k=1 from dense modeling into the sparse dataset. Then, I implemented the CountVectorizer and TfidfVectorizer as before. The accuracy, in particular for TfidfVectorizer, increased significantly to 63% by customizing the stop words.



| | F1   | Accuracy  |
|----------------------|------|-----------|
| TF-IDF    | 0.72 | 0.63 |






