# Importing necessary python libraries
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as mp
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit as split_shuffle

from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.feature_extraction.text import TfidfTransformer as tft

from sklearn.pipeline import Pipeline as pl
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.linear_model import LogisticRegression as lr
from sklearn.svm import LinearSVC as svm
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.ensemble import RandomForestClassifier as rf

# hide warnings
warnings.filterwarnings('ignore')
# plot in 'whitegrid' style
sns.set_style("whitegrid")
# random seed generator
np.random.seed(7)

# TASK 1 : DATASET LOADING
# Read dataset file in csv format
df = pd.read_csv('D:\\Amazon.csv\\amazon.csv')

# dataframe copied data into another dataframe
dataframe = df.copy()

# ghfghdgf

# TASK 2 : SPLITTING DATASET INTO TRAIN DATA AND TEST DATA
print("Before {}".format(len(dataframe)))
After_Operation_Data = dataframe.dropna(subset=["reviews.rating"])
# NAN removing
print("After {}".format(len(After_Operation_Data)))
After_Operation_Data["reviews.rating"] = After_Operation_Data["reviews.rating"].astype(int)

divide = split_shuffle(n_splits=5, test_size=0.2)
for index_for_train, index_for_test in divide.split(After_Operation_Data,
                                           After_Operation_Data["reviews.rating"]):
    classified_train = After_Operation_Data.reindex(index_for_train)
    classified_test = After_Operation_Data.reindex(index_for_test)

# display number of train records
print(len(classified_train))

# percent of review rating in train records
print(classified_train["reviews.rating"].value_counts()/len(classified_train))

# display number of test records
print(len(classified_test))

# percent of review rating in test records
print(classified_test["reviews.rating"].value_counts()/len(classified_test))

# TASK 3 : DATA ANALYSIS (TRAIN DATA)
statements = classified_train.copy()

# exploring asins and name attributes
print(len(statements["name"].unique()), len(statements["asins"].unique()))
print(statements.info())

# grouping to check 1-many relationship between asins and name
print(statements.groupby("asins")["name"].unique())

# product with two ASINs possibility check
distinct_names = statements[statements["asins"] ==
                          "B00L9EPT8O,B01E6AO69U"]["name"].unique()
for title in distinct_names:
    print(title)

print(statements[statements["asins"] == "B00L9EPT8O,B01E6AO69U"]["name"].value_counts())

# average of rating
print(statements["reviews.rating"].mean())
asins_total = statements["asins"].value_counts().index
mp.subplots(2,1,figsize=(16,12))
mp.subplot(2,1,1)
statements["asins"].value_counts().plot(kind="bar", title="ASIN Frequency")
mp.subplot(2,1,2)
sns.pointplot(x="asins", y="reviews.rating", order=asins_total, data=statements)
mp.xticks(rotation=90)
mp.show()
# average of reviews.rating is shown in graph plot

# TASK 4 : CORRELATIONS ANALYSIS
corr_matrix = statements.corr()
print(corr_matrix)

# reviews.ratings with asins analysis
print(statements.info())
total_number = statements["asins"].value_counts().to_frame()
mean_rating = statements.groupby("asins")["reviews.rating"].mean().to_frame()
find_relation = total_number.join(mean_rating)
mp.scatter("asins", "reviews.rating", data=find_relation)
print(find_relation.corr())
# it is concluded that there is no correlation between ASINs and reviews.rating

# TASK 5 : SENTIMENT ANALYSIS
def sentimental_analyzer(rate):
    if (rate == 2) or (rate == 1):  # negative review
        return "Negative"
    elif rate == 3: # neutral review
        return "Neutral"
    elif (rate == 5) or (rate == 4):  # positive review
        return "Positive"
# Adding sentiments
classified_train["Sentiment"] = classified_train["reviews.rating"].apply(sentimental_analyzer)
classified_test["Sentiment"] = classified_test["reviews.rating"].apply(sentimental_analyzer)
print(classified_train["Sentiment"][:20])

# Preparing data
Final_data_train = classified_train["reviews.text"]
Final_data_train_sentiment = classified_train["Sentiment"]
Final_data_test = classified_test["reviews.text"]
Final_data_test_sentiment = classified_test["Sentiment"]

# printing training samples and testing samples
print(len(Final_data_train), len(Final_data_test))

# TASK 6 : FEATURE EXTRACTION
# Replace "nan" with space
Final_data_train = Final_data_train.fillna(' ')
Final_data_test = Final_data_test.fillna(' ')
Final_data_train_sentiment = Final_data_train_sentiment.fillna(' ')
Final_data_test_sentiment = Final_data_test_sentiment.fillna(' ')

# Text pre-processed and counting number of times it occurs
vector_count = cv()
Final_data_train_counts = vector_count.fit_transform(Final_data_train)
print(Final_data_train_counts.shape)

# using TF-IDF Transformer to divide number of occurances for each word
tf = tft(use_idf=False)
Final_data_train_tf = tf.fit_transform(Final_data_train_counts)
print(Final_data_train_tf.shape)

# TASK 7 : MACHINE LEARNING ALGORITHMS

# 7.1 : MULTINOMIAL NAIVE BAYES
Multinomial_NB_Classifier = pl([("vect", cv()),
                             ("tfidf", tft()),
                             ("clf_nominalNB", mnb())])
Multinomial_NB_Classifier.fit(Final_data_train, Final_data_train_sentiment)

Multinomial_NB_Predicted = Multinomial_NB_Classifier.predict(Final_data_test)
print("ACCURACY FOR MULTINOMIAL NAIVE BAYES ALGORITHM : ", np.mean(Multinomial_NB_Predicted == Final_data_test_sentiment))

# 7.2 LOGISTIC REGRESSION

LR_Classifier = pl([("vect", cv()),
                            ("tfidf", tft()),
                            ("clf_logReg", lr())])
LR_Classifier.fit(Final_data_train, Final_data_train_sentiment)

LR_Predicted = LR_Classifier.predict(Final_data_test)
print("ACCURACY FOR LOGISTIC REGRESSION ALGORITHM : ", np.mean(LR_Predicted == Final_data_test_sentiment))

# 7.3 SUPPORT VECTOR MACHINE
SVM_Classifier = pl([("vect", cv()),
                               ("tfidf", tft()),
                               ("clf_linearSVC", svm())])
SVM_Classifier.fit(Final_data_train, Final_data_train_sentiment)

SVM_Predicted = SVM_Classifier.predict(Final_data_test)
print("ACCURACY FOR SUPPORT VECTOR MACHINE ALGORITHM : ", np.mean(SVM_Predicted == Final_data_test_sentiment))

# 7.4 DECISION TREE
DT_Classifier = pl([("vect", cv()),
                                  ("tfidf", tft()),
                                  ("clf_decisionTree", dt())
                                 ])
DT_Classifier.fit(Final_data_train, Final_data_train_sentiment)

DT_Predicted = DT_Classifier.predict(Final_data_test)
print("ACCURACY FOR DECISION TREE ALGORITHM : ", np.mean(DT_Predicted == Final_data_test_sentiment))

# 7.5 RANDOM FOREST
RF_Classifier = pl([("vect", cv()),
                                  ("tfidf", tft()),
                                  ("clf_randomForest", rf())
                                 ])
RF_Classifier.fit(Final_data_train, Final_data_train_sentiment)

RF_Predicted = RF_Classifier.predict(Final_data_test)
print("ACCURACY FOR RANDOM FOREST ALGORITHM : ", np.mean(RF_Predicted == Final_data_test_sentiment))

# TASK 8 : COMPARISON STUDY OF MACHINE LEARNING ALGORITHMS BASED ON PERFORMANCE
# Rounding by 2 decimal points and converting into percent
MNB_Acc = round((np.mean(Multinomial_NB_Predicted == Final_data_test_sentiment) * 100),2)
LR_Acc = round((np.mean(LR_Predicted == Final_data_test_sentiment) * 100), 2)
SVM_Acc = round((np.mean(SVM_Predicted == Final_data_test_sentiment) * 100), 2)
DT_Acc = round((np.mean(DT_Predicted == Final_data_test_sentiment) * 100), 2)
RF_Acc = round((np.mean(RF_Predicted == Final_data_test_sentiment) * 100), 2)

# plot (width, height)
mp.figure(figsize = (12,7))
algorithm = ['Multinomial Naive Bayes', 'Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'Random Forest']
accuracy = [MNB_Acc, LR_Acc, SVM_Acc, DT_Acc, RF_Acc]

# creating bar plot
mp.bar(algorithm, accuracy, width= 0.9, align='center', color='cyan', edgecolor = 'red')
# annotated text location
x = 1.0
y = 0.1

# bar plot annotation with accuracy
for x in range(len(algorithm)):
    mp.annotate(accuracy[x], (-0.1 + x, accuracy[x] + y))

# legend creation
mp.legend(labels = ['Accuracy'])

# plot title
mp.title("Comparison Study of Machine Learning Algorithms Based on Accuracy")

# x and y axis notations
mp.xlabel('MACHINE LEARNING ALGORITHMS')
mp.ylabel('PERFORMANCE')

# plot display
mp.show()
