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
    * Not bad for the first run
