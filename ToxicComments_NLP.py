# TOXIC COMMENTS NATURAL LANGUAGE PROCESSING
# **Name:** Awais Shah

# **Date:** Feb 29 cohort

# --

# ### Problem Statement

# You are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

# toxic  
# severe_toxic  
# obscene  
# threat  
# insult  
# identity_hate   

# You must create a model which predicts a probability of each type of toxicity for each comment. 
# The metric we will be using is ROC AUC 

# Link to raw data



# ##### [https://drive.google.com/drive/folders/13ql-6srXUHyOqTyWrKowPOWjmji-3kqA?usp=sharing](https://drive.google.com/drive/folders/13ql-6srXUHyOqTyWrKowPOWjmji-3kqA?usp=sharing)
# --

# To document my weekly progress in this Kaggle project, I've taken notes in the following Google Doc: https://docs.google.com/document/d/1uqpjUP6hm1SOeIPMIRXsP7SGDG-ssX5j1FdbZoICdZE/edit?usp=sharing

# Import required packages for this notebook.
# """

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.metrics import f1_score,ConfusionMatrixDisplay, classification_report,RocCurveDisplay, precision_recall_curve, accuracy_score, confusion_matrix,precision_score,recall_score,roc_curve,roc_auc_score
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.tree import DecisionTreeRegressor
# import mglearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import _stop_words as stop_words
from sklearn.model_selection import  RepeatedStratifiedKFold
import string
import re
import pandas as pd
# from google.colab import drive
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.ma.core import mean
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN   
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from imblearn.over_sampling import SMOTE
from sklearn.metrics._plot.roc_curve import plot_roc_curve
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import auc

"""This mounts your Google Drive at the location `/content/drive` on the virtual machine running this notebook."""

drive.mount('/content/drive')
filepath = '/content/drive/My Drive/fourthbrain/Kaggle/train.csv'
test_filepath='/content/drive/My Drive/fourthbrain/Kaggle/test.csv'
test_labels_filepath='/content/drive/My Drive/fourthbrain/Kaggle/test_labels.csv'
df=pd.read_csv(filepath)
df_test_comments=pd.read_csv(test_filepath)
df_test_labels=pd.read_csv(test_labels_filepath)
df_test=pd.merge(df_test_comments,df_test_labels,on=['id'])

"""## Load the data"""

pd.set_option('display.max_colwidth',10000)

"""After bringing the data in, we can see what the data looks like. The comment_text column seems to have some grammatical issues that need to be cleaned and we can also see that our Y variable is multiple columns i.e. it is a multilabel classification problem"""

df=pd.read_csv(filepath)
df.head(10)

df.describe()

"""We will now set up a function to clean up the data before sending it through the pipeline for fitting"""

# Let's clean some of the data. As you can see, there are a lot of transfer notations that may cause our model to not perform at its best.
import re
imp=[]
RE_SUSPICIOUS=re.compile(r'[&#<>{}\[\]\\]')
def impurity(text,min_len=10):
  # returns bad text
  if text==None or len(text)<min_len:
    return 0
  else:
    return len(RE_SUSPICIOUS.findall(text))/len(text)

[imp.append(impurity(df["comment_text"][i])) for i,text in enumerate(df["comment_text"])]
  
print(mean(imp))

import html

def clean(text):
    # convert html escapes like &amp; to characters.
    # text = html.unescape(text)
    # # tags like <tab>
    # text = re.sub(r'<[^<>]*>', ' ', text)
    # # markdown URLs like [Some text](https://....)
    # text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r'\1', text)
    # # text or code in brackets like [0]
    # text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    # standalone sequences of hyphens like --- or ==”
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of white spaces
    text = re.sub(r'\s+', ' ', text)
    
    text=text.strip()
    
    return text.strip()

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

df["comment_text"]=df["comment_text"].apply(clean)
df_test["comment_text"]=df_test["comment_text"].apply(clean)

df["comment_text"]=df["comment_text"].apply(decontracted)
df_test["comment_text"]=df_test["comment_text"].apply(decontracted)

mean(df["comment_text"].apply(impurity))

df.head(20)

# df.drop(columns="id", inplace=True)
# df_test.drop(columns="id", inplace=True)

"""## Defining the problem: EDA and Data Pre-processing

We can now look at the count of each class and look for disparities in the dataset that may influence our output metric or our predictions. The skewness that is evident in a majority of the data is leaning toxic, obscene, and insult (in that order); whereas severe_toxic, threat, and identity_hate have minimal occurences. We should note that threat is the worst-case in this data due to its minimal data so we would be using that as our basis in terms of judging performance as we go forward.
"""

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y=np.array(df[categories])
sum=np.sum(y,axis=0)

plots=sns.barplot(categories,sum)
for bar in plots.patches:
  plt.annotate(format(bar.get_height()),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=7, xytext=(0, 8),
                   textcoords='offset points')
plt.xlabel("Category")
plt.ylabel("# of Occurences")

"""We can also look at the multiclass labels for each instance. The one trend we notice is an emphasis on rows where all categories are 0. This is an imbalanced dataset. Let's look at the distribution of the X variable i.e. Comment_text to see any patterns."""

sum=np.sum(df[categories],axis=1)
sum=sum.value_counts().reset_index(name="cnt")
labels =  np.arange(0,7)
# labels
plt.bar(labels,sum["cnt"],color="rgby")

plt.xlabel("Sum of OHE labels")
plt.ylabel("# of Occurences")

"""Threat seems to be the most imbalanced label as shown below in terms of percentage of the total count

"""

for category in categories:
  print("{0:s}: {1:.2f}%".format(category,mean(df[category])*100))

"""The distibution plot shows a skewness in the length of comment_text, where the majority of stays within the range of 0-1000 characters. The max goes all the way to 5000 in certain cases"""

commentlen=df["comment_text"].str.len()
commentlen.hist(bins=np.arange(0,6000,50))

"""Let's check if there are any empty comments


"""

commentleng=df[(df["comment_text"].str.len()==0)]
# below is another way---probably simpler
print("Comments")
print('train',df['comment_text'].isnull().sum())
print('test',df_test['comment_text'].isnull().sum())
# df.drop(commentleng,axis=0,inplace=True)
print("Categories")
print('train',df[categories].isnull().sum())
print('test',df_test[categories].isnull().sum())

df.shape

df_test.shape

"""It is important to convert the text in the comment_text field to a vector format so that the algorithm can process it as numbers. This is why we use TfidVectorizer to clean up the training and test data"""

vect=TfidfVectorizer(stop_words="english")
X_train_dtm=vect.fit_transform(df["comment_text"])
X_test_dtm=vect.transform(df_test["comment_text"])

print("Vectorized feature shapes:\nX_train_dtm: {}\nX_test_dtm: {}".format(X_train_dtm.shape,X_test_dtm.shape))

"""## **BASIC LOGISTIC REGRESSION WITH CV**

Our base model below gives us the ROC AUC values that seem particularly inflated without doing any sort of data adjustment to better the imbalance. This could also be due to the fact that ROC AUC is not a good metric for imbalanced datasets
"""

cv = RepeatedStratifiedKFold(n_repeats=3, random_state=1)

model=LogisticRegression()
mean_roc=[]
for category in categories:
  LR_y_pred = cross_val_predict(model, X_train_dtm, df[category],cv=3, n_jobs=-1)
  mean_roc.append(roc_auc_score(df[category],LR_y_pred))
  print('Mean ROC AUC Score for '+category+': %.3f' % roc_auc_score(df[category],LR_y_pred))
print("Average ROC AUC over all categories is: {0:.2f}".format(np.mean(mean_roc)))

model_balanced=LogisticRegression(class_weight="balanced")
mean_roc_b=[]
for category in categories:
  LR_y_pred_b = cross_val_predict(model_balanced, X_train_dtm, df[category],cv=3, n_jobs=-1)
  mean_roc_b.append(roc_auc_score(df[category],LR_y_pred_b))
  print('Mean ROC AUC Score for '+category+': %.3f' % roc_auc_score(df[category],LR_y_pred_b))
print("Average ROC AUC over all categories is: {0:.2f}".format(np.mean(mean_roc_b)))

model_adj_weights=LogisticRegression(class_weight={0:1,1:300})
mean_roc_adj_weights=[]
for category in categories:
  LR_y_pred_adj_weights = cross_val_predict(model_adj_weights, X_train_dtm, df[category],cv=3, n_jobs=-1)
  mean_roc_adj_weights.append(roc_auc_score(df[category],LR_y_pred_adj_weights))
  print('Mean ROC AUC Score for '+category+': %.3f' % roc_auc_score(df[category],LR_y_pred_adj_weights))
print("Average ROC AUC over all categories is: {0:.2f}".format(np.mean(mean_roc_adj_weights)))

steps = [('over', SMOTE()), ('model', LogisticRegression())]
pipeline = Pipeline(steps=steps)
mean_roc_smote=[]
for category in categories:
  LR_y_pred_smote = cross_val_predict(pipeline, X_train_dtm, df[category],cv=3, n_jobs=-1)
  mean_roc_smote.append(roc_auc_score(df[category],LR_y_pred_smote))
  print('Mean ROC AUC Score for '+category+': %.3f' % roc_auc_score(df[category],LR_y_pred_smote))
print("Average ROC AUC over all categories is: {0:.2f}".format(np.mean(mean_roc_smote)))

"""## **MULTILABEL KNEIGHBORS CLASSIFICATION**

We move on to KNeighborsClassifier with a multilabel average-macro. Macro just means that each of the labels hold equal importance. As you can see the f1 and roc auc scores are not impressive. So it does give
"""

from sklearn.neighbors import KNeighborsClassifier
y_multilabel=df[categories]
knn_clf=KNeighborsClassifier()
# knn_clf.fit(X_train_dtm,y_multilabel)
y_train_pred_kn=cross_val_predict(knn_clf,X_train_dtm,y_multilabel,cv=3)
print("roc auc",roc_auc_score(y_multilabel,y_train_pred_kn,average="macro"))

"""## **DECISION TREE CLASSIFICATION WITH SMOTE**

Below we use a decision tree classifier with SMOTE to adjust the imbalance.
"""

from sklearn.metrics._plot.roc_curve import plot_roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import auc

for category in categories:
  DT_y_pred = cross_val_predict(DecisionTreeClassifier(class_weight='balanced'), X_train_dtm, df[category], method='predict_proba',cv=3,  n_jobs=-1)
  y_scores_forest=DT_y_pred[:,1]
  fpr_forest,tpr_forest,threshold_forest=roc_curve(df[category],y_scores_forest)
  ac=auc(fpr_forest,tpr_forest)
  print("AUC for {0:s}: {1:.2f}".format(category,ac))
#   disp=RocCurveDisplay(fpr=fpr_forest,tpr=tpr_forest,roc_auc=ac,)
#   disp.plot()
# plt.legend()
# plt.show()

"""## **GRADIENT BOOSTING WITH XGBOOST AND STRATIFIED K-FOLD CV**

We can try XGBoost with StratifiedKFold CV to ensure a good distribution of imbalanced labels. Using XGBoost will also allow for a sequential gradient boosting and fixing of errors to get the best accuracy.
"""

import xgboost
kfold = StratifiedKFold(n_splits=5)
# fit model no training data
model = xgboost.XGBClassifier()
for category in categories:
  results = cross_val_score(model, X_train_dtm, df[category], cv=kfold)
  print(category)
  print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# classifiers={
#     "NaiveBayes": MultinomialNB(),
#     "LogisticRegression": LogisticRegression(max_iter=1000,class_weight={0:1,1:100},random_state=42),
#     "BernoulliNaiveBayes": BernoulliNB(),
#     "LinearSVC": LinearSVC(random_state=42) 
# }

# f, axes = plt.subplots(4, 6, figsize=(40, 10), sharey='row')
# for i, (key, classifier) in enumerate(classifiers.items()):
#   steps = [('over', SMOTE()), ('model', classifier)]
#   pipeline = Pipeline(steps=steps)
#   print(classifier)
#   for x,category in enumerate(categories):
#     pipeline.fit(X_train_dtm,df[category])
#     classifier_y_pred=pipeline.predict(X_test_dtm)
#     cm=confusion_matrix(df_test[category],classifier_y_pred)    
#     disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier.classes_)
#     disp.plot(ax=axes[i,x], xticks_rotation=45)
#     disp.ax_.set_title(category)
#     disp.im_.colorbar.remove()
#     disp.ax_.set_xlabel('')
#     if i!=0:
#         disp.ax_.set_ylabel('')
#     print(category)
#     print("f1",f1_score(df_test[category],classifier_y_pred,average="macro"))
#     # print("roc auc",roc_auc_score(df_test[category],classifier_y_pred))
#     print('precision',precision_score(df_test[category],classifier_y_pred,average="macro"))
#     print('recall',recall_score(df_test[category],classifier_y_pred,average="macro"))
# f.text(0.6, 0.4, 'Predicted label', ha='left')
# plt.subplots_adjust(wspace=0.60, hspace=0.4)


# f.colorbar(disp.im_, ax=axes)
# plt.show()

"""The above confusion matrix for Naive Bayes and

## **NAIVE BAYES with MULTILABELS**
"""

steps = [('over', SMOTE()), ('model', MultinomialNB())]
cv = StratifiedKFold()
NB_model = Pipeline(steps=steps)
f, axes = plt.subplots(1, 6, figsize=(20, 5), sharey='row')
for i,category in enumerate(categories):
  NB_y_pred = cross_val_predict(NB_model, X_train_dtm, df[category], cv=cv, n_jobs=-1)
  # NB_model.fit(X_train_dtm,train[category])
  # NB_y_pred=NB_model.predict(X_test_dtm)
  cm=confusion_matrix(df[category],NB_y_pred)
  disp=ConfusionMatrixDisplay(confusion_matrix=cm)
  disp.plot(ax=axes[i], xticks_rotation=45)
  disp.ax_.set_title(category)
  disp.im_.colorbar.remove()
  disp.ax_.set_xlabel('')
  if i!=0:
      disp.ax_.set_ylabel('')
  print(category)
  # print("f1",f1_score(df[category],NB_y_pred,average="macro"))
  print("roc auc",roc_auc_score(df[category],NB_y_pred,average="macro"))
  print('precision',precision_score(df[category],NB_y_pred,average="macro"))
  print('recall',recall_score(df[category],NB_y_pred,average="macro"))
f.text(0.4, 0.1, 'Predicted label', ha='left')
plt.subplots_adjust(wspace=0.60, hspace=0.4)


f.colorbar(disp.im_, ax=axes)
plt.show()

"""That is a lot of features, lets try to set min_df to ensure appearance of words in a document at least 5 times for it to be considered for a feature

It barely made a dent. Two options from here, either try max_df or rescaling the data with tf-idf (term frequency- inverse document frequency). Lets try rescaling. The idea behind this is to weight often appearing terms heavily within a document, but not in many documents in teh corpus. Se essentially if a word reappears a lot within a document but not across documents then its a descriptor of the content. Scikit implements this in 2 classes: 1)TfidfTransformer, which takens in the matrix(sparse) output of CountVectorizer and then transforms it, and 2) TfidfVectorizer, which does both bag-of-words feature extraction and tf-idf transformation. Both classes also apply L2 normalization i.e. rescaling the representation of each docuemnt to have euclidean norm 1. So essentially, if the number of words change, the vectorized representation would remain the same

We can inspect which words were found be most important (Note: this is unsupervised technique so that idea might be a little subjective)

In Naive Bayes, we have two options for discrete data i.e. Bernoulli and Multinomial. We use Multinomial below to see how it fits the data. Let's run the code and see the scores.

Likely underfitting with logistic since training and testing results are so close

## **NN**
"""

#Testing Deep learning
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
max_features=20000
maxlen=100
training_samples=0.8*len(df)
print(len(df))
validation_samples=0.2*len(df)
max_words=10000
# X_train=X_train.reshape(len(X_train),1)
tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df["comment_text"])
sequences=tokenizer.texts_to_sequences(df["comment_text"])
word_index=tokenizer.word_index
print("Found %s unique tokens." % len(word_index))
data=pad_sequences(sequences,maxlen=maxlen)

labels=np.asarray(df[categories])
print("Shape of data tensor:", data.shape)
print("Shape of label tensor:", labels.shape)
indices=np.arange(data.shape[0])
np.random.shuffle(indices)
data=data[indices]
labels=labels[indices]
X_train,X_val=train_test_split(data, test_size=0.2, shuffle=True, random_state=42)
y_train,y_val=train_test_split(labels, test_size=0.2, shuffle=True, random_state=42)
# y_train=labels[:training_samples]
# X_val=data[training_samples: training_samples+validation_samples]
# y_val=data[training_samples: training_samples+validation_samples]
# X_train=preprocessing.sequence.pad_sequences(X_train,maxlen=maxlen)
# X_val=preprocessing.sequence.pad_sequences(X_val,maxlen=maxlen)
import os
embedding_filepath='/content/drive/My Drive/Datasets/Kaggle/WordEmbeddings'
embeddings_index={}

f=open(os.path.join(embedding_filepath,"glove.6B.100d.txt"))
for line in f:
  values=line.split()
  word=values[0]
  coefs=np.asarray(values[1:],dtype="float32")
  embeddings_index[word]=coefs
f.close()
print("Found %s word vectors." % len(embeddings_index))
embedding_dim=100
embedding_matrix=np.zeros((max_words,embedding_dim))
for word,i in word_index.items():
  if i<max_words:
    embedding_vector=embeddings_index.get(word)
    if embedding_vector is not None:
      embedding_matrix[i]= embedding_vector
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model=Sequential()
model.add(Embedding(max_words,embedding_dim,input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable=False
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["acc"])
history=model.fit(X_train,y_train[:,3],epochs=10, batch_size=32, validation_data=(X_val,y_val[:,3]))
model.save_weights("pre_trained_glove_model.h5")
acc=history.history["acc"]
val_acc=history.history["val_acc"]
loss=history.history["loss"]
val_loss=history.history["val_loss"]

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,"bo", label="Training acc")
plt.plot(epochs,val_acc,"b", label="Validation acc")
plt.title("Training and Validation accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

model=Sequential()
model.add(Embedding(max_words,embedding_dim,input_length=maxlen))
model.add(Flatten())
model.add(Dense(32,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.summary()
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["acc"])
history=model.fit(X_train,y_train[:,3],epochs=10,batch_size=32,validation_data=(X_val,y_val[:,3]))

model.save_weights("pre_trained_glove_model.h5")
acc=history.history["acc"]
val_acc=history.history["val_acc"]
loss=history.history["loss"]
val_loss=history.history["val_loss"]

epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,"bo", label="Training acc")
plt.plot(epochs,val_acc,"b", label="Validation acc")
plt.title("Training and Validation accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()