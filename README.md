# SENTIMENT-ANALYSIS-WITH-NLP

COMPANY: CODTECH IT SOLUTIONS

NAME: VITHANALA POOJITHA

INTERN ID: CTO4DY196

DOMAIN: MACHINE LEARNING

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

Task 2: Sentiment Analysis with NLP – Detailed Description

The second internship task is conducting Sentiment Analysis on a customer review dataset using Natural Language Processing (NLP) methods. The goal is to process textual data, extract useful patterns, and categorize sentiments like positive or negative. This assignment integrates prominent concepts in text preprocessing, feature extraction with TF-IDF, model construction with Logistic Regression and result evaluation.

Introduction to Sentiment Analysis
Sentiment Analysis is the process of identifying whether a block of text carries a positive, negative, or neutral sentiment. Sentiment analysis has numerous applications across industries such as e-commerce, social media monitoring, product reviews, and customer feedback analysis. Businesses utilize it to gauge customer satisfaction, identify product problems, and enhance user experience. In this exercise, we want to develop a basic yet efficient sentiment analysis model through machine learning.

Objectives of the Task
1. Preprocess raw text data and get it ready for analysis.
2. Extract features from text through **TF-IDF vectorization**.
3. Train a **Logistic Regression** classifier for sentiment classification.
4. Assess the model's performance through accuracy, classification reports, and confusion matrix.
5. Visualize and explain the results.

Steps Involved
1. Data Loading and Exploration
The data set usually has customer reviews (text) and labels (sentiments) corresponding to it. For instance, a review like "The product was excellent!" would be labeled Positive, whereas "The service was terrible." would be labeled Negative. Browsing through the data set allows us to realize the sentiment distribution as well as identify faults such as missing values.

2. Text Preprocessing
Because raw text data usually has noise, it must be preprocessed before it can be incorporated into a machine learning model. Typical steps in preprocessing are:

Lowercasing: Converting words into lowercase to ensure consistency.
Removing Punctuation & Special Characters: Removing unnecessary symbols that do not add to sentiment.
Tokenization: Breaking text down into words (tokens).
Stopword Removal: Deleting common words such as "is," "the," "and" that do not have sentiment.
Stemming/Lemmatization: Condensing words into their base forms (e.g., "running" → "run").

Following preprocessing, a new column like `\"Cleaned_Review\"` is added for the purpose of storing cleaned text.

3. Feature Extraction using TF-IDF
Machine learning models cannot directly process raw text. Therefore, text needs to be mapped into numerical features. Here we employ **TF-IDF (Term Frequency – Inverse Document Frequency)**, a method that weighs words according to their frequency of appearance within a document relative to the whole dataset. High-frequency words in one document but low-frequency words throughout the others are weighted more. This helps significant words such as "amazing," "bad," "disappointing," or "excellent" have greater impact on the model than ubiquitous words such as "the" or "and."

4. Splitting the Data
The data is divided into **training** and **testing** sets. Training data are used to train the model, and test data are used to test its performance on unseen examples. It is usually 75% for training and 25% for testing.

5. Model Building with Logistic Regression
To perform this task, we employ **Logistic Regression**, a straightforward yet effective classification algorithm. Logistic Regression predicts the probability that a specific input is of a specific class. In sentiment analysis, it classifies a review as either **positive** or **negative**. Although it is a linear model, Logistic Regression tends to work extremely well on text classification tasks with TF-IDF features.

6. Model Training and Prediction
The Logistic Regression model is trained over TF-IDF transformed training data. After training, the model predicts sentiments for the test dataset. Predictions are matched against actual labels to measure performance.

7. Model Evaluation
Model performance is measured by:
Accuracy Score:The ratio of correct predictions.
Classification Report: Provides precision, recall, and F1-score per class.
Confusion Matrix: A table visualization that indicates the number of correct or misclassified predictions. For instance, how many wrongly predicted negative reviews were there.

Conclusion
Real-world experience in text data handling and machine learning-based classification. It addresses the complete pipeline of NLP operations from text preprocessing to feature extraction, model construction, and evaluation. One learns essential skills in converting raw text into insightful information by working on this task, which is essential for actual usage in customer review analysis, brand tracking, and decision-making.

This exercise not only illustrates the usefulness of preprocessing and feature engineering but also illustrates how naive models such as Logistic Regression can produce strong results when used appropriately to text data.


OUTPUT

<img width="763" height="637" alt="Image" src="https://github.com/user-attachments/assets/02e19557-8960-4758-9fef-af5cbb1a8416" />
