# :right_anger_bubble:Toxic Comments Classification in Natural Language Processing

This project creates a classifier that is able to categorize comments into 6 categories i.e. Toxic, Severe Toxic, Threat, Insult, Obscene, and Identity Hate. We will be using multiple ML methods as a means of learning how to apply those ML methods and also to find out the best methods. 
The kaggle dataset can be found [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

## Table of Contents
The following topics will be covered:

1\. Part 1: Core ML Classification Methods
  - Basic Logistic Regression with CV
  - MultiLabel K-Neighbors Classification
  - Decision Tree Classification
  - Gradient Boosting with XGBoost and Stratified K-Fold CV
  - Naive Bayes with Multilabels

2\. Part 2: Deep Learning for Toxic Text Classification
  - Binary N-Gram with TF-IDF Vectorization
  - Bi-Directional LSTM
  - GLOVE Pre-Trained Word Embeddings
  - Transformers with Positional encoding in Keras
3\. Conclusion


### Tools Used:

* Tensorflow 2.3
* Python 3.8
* Google Colab


## Part 1 Takeaways

* Requires some text preprocessing
* After some EDA, we realize the data is extremely imbalanced. Our models work on finding ways to overcome that problem.
* The best method in this sections seems to be a Logistic Regression based on ROC AUC.

## Part 2 Takeaways

* Text preprocessing easily managed
* Text embeddings are great if you have small amount of data
* Difference between N-Gram models and Sequence models

## Project Takeaway

* Occam's razor: Sometimes the simplest method with minimal complexity is the best method


## Authors

* **Awais Shah** 

## Acknowledgments and Links used

* [FChollet- Deep Learning with Python](https://github.com/fchollet/deep-learning-with-python-notebooks)
* [Geron-Hands-On ML v2](https://github.com/ageron/handson-ml2)
* [:hugs:](https://huggingface.co/)
* [sklearn-It is a gold mine of ML concepts](https://scikit-learn.org/stable/user_guide.html)
