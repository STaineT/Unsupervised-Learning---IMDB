# Unsupervised Sentiment Analysis - Trustpage Challenge
### Understanding the problem
* Classification problem – determining a binary sentiment variable<br>
* What ML algorithms to solve the problem<br>
* The goal is to detect the positive and negative reviews<br>
* The measure of success can be accuracy or the area under the curve<br>
* KMean clustering was selected – define the number of clusters that the <br> algorithm strives to determine – in this problem 2, for positive and negative reviews

### Inspecting the data set and preprocessing
* 50,000 rows in the data, review – sentiment<br>
* Explored the data<br>
* The data is text – thus have to use NLP to build a meaningful dataset<br>
    * Extracting features from movie reviews with the bag of word technique<br> 
    * Bag of words analyze the entire review data sets, builds a dictionary of all words, and creates a list of numbers for each word, and counts how many times each word appears in the dataset<br>
* Created a function to clean the data, including:<br>
    * removing HTML tags<br>
    * removing special characters<br>
    * expanding the abbreviate forms<br>
    *  …
    
### Basic Modeling
* decided to create a simple working version of KMenas<br>
* I vectorize the data set – I used CountVectorizer from sklearn.feature_extraction.text<br>
* The resulting array is a sparse array – Sparse dataset<br> 
* Create an initial model with default hyperparameters<br>
* And then used PCA to reduce the dimensionality of the dataset<br> 
* The results for PCA was slightly better with<br> 
    * <font color=blue>F1 Score : 0.62 and Accuracy 0.55</font>
* Trying to improve the model performance – I used TfidVectorizer from sklearn.feature_extraction.text<br>
    * <font color=blue>F1 Score : 0.34 and Accuracy 0.45</font>
* TfidVectorizer performed substantially better without PCA

### Tuning the parameters for modeling
* in order to improve the performance of the model, a combination of three different values for each parameter were tested, a total number of 27 models for each case
* max_fatures is the maximum number of world columns – 10000, 30000, None
* min_df minimum number of times a word must appear in the text – 1, 2, 3
* max_iter maximum number of iteration for KMenas – 1000, 2000, 3000
    * tuning practically had no impact on CountVectorizer
    * tuning practically had little impact on TfidVectorizer with increasing the accuracy to 57%
### Selecting TfidVectorizer with max_feature=None, min_df=2, max_iter=1000
* visualized the data. The performance is not great
* moving to the model which is using KMeans on Dense 

### Basic Modeling – Wor2Vec
* highest accuracy achieved with tuning the parameters with sparse dataset was 57%
* Word2Vec is a group of related models that are used to produce word embeddings
* Word2Vec is not a single algorithm. It consists of two algorithms GBOW and Skip-gram
* applied the basic modeling with default values using Wor2Vec and then KMeans clustering again with two clusters
* Achieved 62% accuracy using the dense matrix, improved significantly over sparse datasets
### Tuning the model – Word2Vec
* train the model for 30 epochs
* changed the text processing to include the bigrams as well
* accuracy dropped to 52%  
