# INTEL Products Sentiment Analysis from Online Reviews 
PS-11 Intel Products Sentiment Analysis from Online Reviews

# PROBLEM STATEMENT
Customer feedback and Reviews plays a crucial role in the continuous improvement of products and services. Companies often receive vast amounts of textual reviews from end users and tech reviewers on various platforms, making it challenging to manually analyse and derive actionable insights from this data.

Our project aims to address this issue by developing an automated sentiment analysis system that can process customer reviews, detect trends over time, and provide recommendations for INTEL Products (MOBILE/DESKTOP PROCESSOR) using a chatbot

# SCOPE OF THE PROJECT
The project aims to analyze customer sentiment towards : 

12th generation ( mobile / desktop processor )

13th generation ( mobile / desktop processor )

14th generation ( mobile / desktop processor )

The user end reviews were taken from the 

AMAZON WEBSITE (https://www.amazon.in) around the time frames of 2022 to 2024.

The tech reviews were taken from 3 websites :

PC MAG : https://www.pcmag.com/categories/processors/brands/intel

AnandTech : https://www.anandtech.com/show/18740/the-intel-core-i3-13100f-review-finding-value-in-intels-cheapest-chip

Trusted Reviews : https://www.trustedreviews.com/best/best-intel-processor-3517396


# DATA SOURCE 
End user reviews : AMAZON (online platform)
Websites like Amazon offer INTEL product reviews from customers which we used for sentiment analysis. 

Tech reviews :  PC MAG , AnandTECH , TRUSTED REVIEWS 
The above websites offered us technical reviews of INTEL products which we used for sentimental analysis

# DATA COLLECTION METHODS
WEB SCRAPING :  
Data was collected from Amazon product reviews using web scraping techniques. The scraping was performed using Python scripts to extract user reviews, including the review text, rating, and date of submission. The collected data underwent cleaning to remove HTML tags, URLs, and special characters, followed by normalization, tokenization, and removal of stop words

# DATA PREPROCESSING  :
The dataset is loaded and pre-processed to handle missing values and convert non-numeric values to numeric ones.
Data Cleaning:
•	Dropped unnecessary columns.

•	Removed duplicate entries.

•	Extracted numeric ratings from the 'rating' column.

•	Filled missing values in the 'review' column with empty strings.

•	Added a new column 'review_length' to store the length of each review.

•	Tokenization: Splitting of review text into individual words or tokens.

•	Stop Words Removal: Elimination of common words that do not contribute to sentiment analysis (e.g., "and," "the")

# SENTIMENT ANALYSIS METHODOLOGY :
 # SENTIMENT ANALYSIS APPROACH : 
1.	Rule based approach
2.	Machine Learning 
3.	Deep Learning
   
 # MODEL SELECTION :
1.	Vader Model 
2.	Roberta Model
3.	Random Forest – Highest Accuracy Amongst Other Ml Model
4.	LSTM – (DL)

# FEATURE EXTRACTION: 
Feature extraction is the process of transforming raw data into a set of features that can be effectively used for machine learning tasks.
1.	Text Preprocessing:
2.	Vectorization : Bag of Words (BoW) , TF-IDF (Term Frequency-Inverse Document Frequency)
3.	Contextual Embeddings: Utilizing models like BERT, RoBERTa
4.	Feature Selection: Sentiment Lexicons , Part-of-Speech (POS) Tagging

# IMPLEMENTATION
TOOLS AND LIBRARIES

•	Pandas

•	NumPy

•	Scikit-Learn

•	TensorFlow/Keras

•	NLTK (Natural Language Toolkit)

•	Gensim

•	Matplotlib

•	Seaborn

•	WordCloud

•	VADER (Valence Aware Dictionary and sEntiment Reasoner)

•	Transformers (Hugging Face)

# MODEL TRAINING 
# 1 LSTM :
•	Imports library and data preparation

•	Model Architecture - An LSTM model is defined with an embedding layer, LSTM layer, and dense layers

•	Model Compilation - The model is compiled with an optimizer , a loss function and evaluation metrics

•	TRAINING THE MODEL – 
•	The model is trained on the prepared dataset using a specified number of epochs and batch size. The dataset is split into training and validation sets to monitor the model's performance during training.

•	EVALUATION – 
•	After training, the model's performance is evaluated on the validation set using accuracy, precision, recall, and F1 score

# 2 RANDOM FOREST :
•	Model Initialization – RandomForestClassifier

•	DEFINING HYPERPARAMETER SPACE - 

1.	n_estimators: Number of trees in the forest.
   
3.	max_depth: Maximum depth of the tree.
   
5.	min_samples_split: Minimum number of samples required to split an internal node.
   
7.	min_samples_leaf: Minimum number of samples required to be at a leaf node. 

•	Randomized Search for Hyperparameter Tuning - RandomizedSearchCV object to perform the search over the hyperparameter space

•	Fitting the Model

•	Best Model Selection – random search

•	Making Predictions and Evaluating Model Performance - printed the accuracy score.

# 3 ROBERTA : 
Model Initialization:
•	The RoBERTa model (TFRobertaForSequenceClassification) and tokenizer (AutoTokenizer) are loaded from the pretrained model cardiffnlp/twitter-roberta-base-sentiment.

Sentiment Analysis Setup:
•	The (SentimentIntensityAnalyzer) from the NLTK library is also initialized to provide an additional sentiment scoring mechanism, ensuring robustness in the sentiment analysis approach

Text Preprocessing:
•	Text data is tokenized using the RoBERTa tokenizer to convert it into a format suitable for the model

# 4 CHATBOT : 
Using the Gemini API, we have created a chatbot that can respond to inquiries regarding Intel's 12th, 13th, and 14th generation processors. The chatbot has access to detailed data on a variety of topics, including user experiences, sentimental patterns from online reviews, pricing, and descriptions. This extensive dataset allows users to ask a variety of topics and get insightful answers from the chatbot.
Features of the Chatbot:
1.	Detailed Descriptions: The chatbot provides in-depth information about each Intel processor generation, highlighting their specifications and key features.
2.	User Experience Insights: By analyzing user reviews, the chatbot offers insights into common user experiences and satisfaction levels.
3.	Pricing Information: Users can inquire about the current pricing and historical price trends of these processors.
4.	Sentiment Analysis: The chatbot uses sentiment analysis to convey general sentiment trends from user reviews, helping users understand the overall performance of each processor generation.
5.	Recommendations: Based on user queries and data trends, the chatbot can suggest the recommendations to improve the performance of the processor.

# ARCHITECTURE DIAGRAM
![image](https://github.com/user-attachments/assets/20ed72f8-aa1c-459f-8b74-8d3b7749aba7)

# CHATBOT - OUTPUTS 

![image](https://github.com/user-attachments/assets/6a3266bb-4c6d-42bd-ac0f-44a213c30def)
![image](https://github.com/user-attachments/assets/5b799c3b-334a-4d79-bcd2-0af2f33f5a62)
![Screenshot (380)](https://github.com/user-attachments/assets/4a7cc827-88f8-4bba-9524-f7e89aea5ec6)




# RESULTS AND DISCUSSION

# 12th GENERATION :
1.	RANDOM FOREST : 82%
2.	XG BOOST : 61%
3.	LOGISTIC REGRESSION : 63%
4.	SVM : 54 %
5.	CNN : 48%
6.	LSTM : 55%
   
# 13th GENERATION :   
1.	RANDOM FOREST : 97%
2.	XG BOOST : 46%
3.	LOGISTIC REGRESSION : 50%
4.	SVM : 54%
5.	CNN : 45%
6.	LSTM : 50%
   
# 14th GENERATION :
1.	RANDOM FOREST : 82%
2.	XG BOOST : 61%
3.	LOGISTIC REGRESSION : 63%
4.	SVM : 54%
5.	CNN : 48%
6.	LSTM : 48%

   
The evaluation of different machine learning models on the 12th, 13th, and 14th generation Intel processors reveals that the Random Forest classifier consistently outperforms other models, achieving the highest accuracy rates of 82% for the 12th and 14th generations and an impressive 97% for the 13th generation. Other models such as XGBoost, Logistic Regression, SVM, CNN, and LSTM show varied performance, with XGBoost and Logistic Regression generally performing better than SVM, CNN, and LSTM. The significant drop in accuracy for most models on the 13th generation, except for Random Forest, indicates potential variability in the dataset or model-specific performance issues. The chatbot leveraging this data provides users with comprehensive insights, ranging from detailed processor descriptions to sentiment analysis based on user reviews. These findings suggest that the Random Forest model is the most reliable for predicting user sentiment and performance trends across different Intel processor generations.

# SUMMARY
This sentiment analysis project involved scraping Amazon product reviews, preprocessing the text data, and training an LSTM model to classify sentiments as Positive, Neutral, or Negative. The main findings indicate that the model accurately identifies sentiment trends, revealing valuable insights into customer opinions and product performance. These implications suggest potential enhancements in customer feedback analysis, helping businesses improve products and services based on user sentiment. 











