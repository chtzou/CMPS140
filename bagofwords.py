#code borrowed from: https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/ch04.html
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold, GridSearchCV
from sklearn.utils import shuffle
from sklearn.feature_extraction import text
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.svm import SVC
import numpy as np

bestModel = "" 
bestModelScore = 0

dfOG = pd.read_csv('train.csv', usecols=['question_text', 'target'])
bad = dfOG[dfOG.target == 1]
df = dfOG[:int(len(dfOG)*.8)] 
dfUnbalanced = dfOG[int(len(dfOG)*.8):] #This is to split in different parts so we have two completely separate tests to test unbalanced vs balanced accuracy
goodQ = df[df.target == 0]
badQ = df[df.target == 1]
goodQ2 = goodQ.sample(len(badQ))

betterTrainingSet = shuffle(pd.concat([badQ, goodQ2]))

training_data_base, test_data_base = train_test_split(dfUnbalanced, train_size=0.7)

bagOfWords = text.CountVectorizer(ngram_range=(1, 2))

xTestUnbalanced = bagOfWords.fit_transform(dfUnbalanced['question_text'])
yTestUnbalanced = dfUnbalanced['target']

#SGD
training_data = betterTrainingSet
xTrain = bagOfWords.transform(training_data['question_text'])
yTrain = training_data['target']

model = SGDClassifier().fit(xTrain, yTrain)
score = model.score(xTestUnbalanced, yTestUnbalanced)
predict = model.predict(xTestUnbalanced)
balancedScore = metrics.balanced_accuracy_score(yTestUnbalanced, predict)

print("SGD: score:", score, "f1:", metrics.f1_score(yTestUnbalanced, predict), "f1 macro:", metrics.f1_score(yTestUnbalanced, predict, average="macro"), "f1 micro:", metrics.f1_score(yTestUnbalanced, predict, average="micro"), "f1 weighted:", metrics.f1_score(yTestUnbalanced, predict, average="weighted"), "f1 pos:", metrics.f1_score(yTestUnbalanced, predict, pos_label=0, average="binary"), "f1 neg:", metrics.f1_score(yTestUnbalanced, predict, pos_label=1, average="binary"),  "precision:",
      metrics.precision_score(yTestUnbalanced, predict), "recall:", metrics.recall_score(yTestUnbalanced, predict), "balanced score:", metrics.balanced_accuracy_score(yTestUnbalanced, predict), "confusion matrix:", metrics.confusion_matrix(yTestUnbalanced, predict))

if bestModelScore < balancedScore:
    bestModelScore = balancedScore
    bestModel = "Pure Logistic Regression"

#Dummy Classifier
dummyTrain, dummyTest = train_test_split(dfOG, train_size=0.7)
xDummy = bagOfWords.transform(dummyTrain['question_text'])
yDummy = dummyTrain['target']
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(xDummy, yDummy)

score = dummy.score(xTestUnbalanced, yTestUnbalanced)
predict = dummy.predict(xTestUnbalanced)
balancedScore = metrics.balanced_accuracy_score(yTestUnbalanced, predict)

print("Dummy Classifier: score:", score, "f1:", metrics.f1_score(yTestUnbalanced, predict), "f1 macro:", metrics.f1_score(yTestUnbalanced, predict, average="macro"), "f1 micro:", metrics.f1_score(yTestUnbalanced, predict, average="micro"), "f1 weighted:", metrics.f1_score(yTestUnbalanced, predict, average="weighted"), "f1 pos:", metrics.f1_score(yTestUnbalanced, predict, pos_label=0, average="binary"), "f1 neg:", metrics.f1_score(yTestUnbalanced, predict, pos_label=1, average="binary"), "precision:",
    metrics.precision_score(yTestUnbalanced, predict), "recall:", metrics.recall_score(yTestUnbalanced, predict), "balanced score:", balancedScore, "confusion matrix:", metrics.confusion_matrix(yTestUnbalanced, predict))

if bestModelScore < balancedScore:
    bestModelScore = balancedScore
    bestModel = "dummy"

#Pure Logistic Regression Model
training_data = betterTrainingSet
xTrain = bagOfWords.transform(training_data['question_text'])
yTrain = training_data['target']

model = LogisticRegression().fit(xTrain, yTrain)
score = model.score(xTestUnbalanced, yTestUnbalanced)
predict = model.predict(xTestUnbalanced)
balancedScore = metrics.balanced_accuracy_score(yTestUnbalanced, predict)

print("Pure Logistic Regression: score:", score, "f1:", metrics.f1_score(yTestUnbalanced, predict), "f1 macro:", metrics.f1_score(yTestUnbalanced, predict, average="macro"), "f1 micro:", metrics.f1_score(yTestUnbalanced, predict, average="micro"), "f1 weighted:", metrics.f1_score(yTestUnbalanced, predict, average="weighted"), "f1 pos:", metrics.f1_score(yTestUnbalanced, predict, pos_label=0, average="binary"), "f1 neg:", metrics.f1_score(yTestUnbalanced, predict, pos_label=1, average="binary"),  "precision:",
    metrics.precision_score(yTestUnbalanced, predict), "recall:", metrics.recall_score(yTestUnbalanced, predict), "balanced score:", metrics.balanced_accuracy_score(yTestUnbalanced, predict), "confusion matrix:", metrics.confusion_matrix(yTestUnbalanced, predict))

if bestModelScore < balancedScore:
    bestModelScore = balancedScore
    bestModel = "Pure Logistic Regression"

#Balanced Logistic Regression
trainUnbalanced, testUnbalanced = train_test_split(
    dfOG, train_size=0.8)

xTrain = bagOfWords.transform(trainUnbalanced['question_text'])
xTest = bagOfWords.transform(testUnbalanced['question_text'])

yTrain = trainUnbalanced['target']
yTest = testUnbalanced['target']

balancedModel = LogisticRegression(class_weight='balanced').fit(
    xTrain, yTrain)
score = balancedModel.score(xTest, yTest)
predict = balancedModel.predict(xTest)
balancedScore = metrics.balanced_accuracy_score(yTest, predict)

print("Balanced: score:", score, "f1:", metrics.f1_score(yTest, predict), "f1 macro:", metrics.f1_score(yTest, predict, average="macro"), "f1 micro:", metrics.f1_score(yTest, predict, average="micro"), "f1 weighted:", metrics.f1_score(yTest, predict, average="weighted"), "f1 pos:", metrics.f1_score(yTest, predict, pos_label=0, average="binary"), "f1 neg:", metrics.f1_score(yTest, predict, pos_label=1, average="binary"), "precision:", metrics.precision_score(yTest, predict), "recall:", metrics.recall_score(
    yTest, predict), "balanced score:", balancedScore, "confusion matrix:", metrics.confusion_matrix(yTest, predict))

if bestModelScore < balancedScore:
    bestModelScore = balancedScore
    bestModel = "Balanced Logistic Regression"

#KFold Logistic Regression Model
kfold = KFold(10, True, 1)

X = betterTrainingSet['question_text']
Y = betterTrainingSet['target']

for trainIndex, testIndex in kfold.split(X):
    xTrain, xTest = X.iloc[trainIndex], X.iloc[testIndex]
    yTrain, yTest = Y.iloc[trainIndex], Y.iloc[testIndex]
    xTrain = bagOfWords.transform(xTrain)
    xTest = bagOfWords.transform(xTest)

    model = LogisticRegression().fit(xTrain, yTrain)
    score1 = model.score(xTestUnbalanced, yTestUnbalanced)
    score = model.score(xTest, yTest)
    predict = model.predict(xTestUnbalanced)
    balancedScore = metrics.balanced_accuracy_score(yTestUnbalanced, predict)

    print("K-Fold Logistic Regression: Unbalanced Test:", score1, "Balanced Test:", score, "f1:", metrics.f1_score(yTestUnbalanced, predict), "f1 macro:", metrics.f1_score(yTestUnbalanced, predict, average="macro"), "f1 micro:", metrics.f1_score(yTestUnbalanced, predict, average="micro"), "f1 weighted:", metrics.f1_score(yTestUnbalanced, predict, average="weighted"), "f1 pos:", metrics.f1_score(yTestUnbalanced, predict, pos_label=0, average="binary"), "f1 neg:", metrics.f1_score(yTestUnbalanced, predict, pos_label=1, average="binary"),  "precision:",
          metrics.precision_score(yTestUnbalanced, predict), "recall:", metrics.recall_score(yTestUnbalanced, predict), "balanced score:", balancedScore, "confusion matrix:", metrics.confusion_matrix(yTestUnbalanced, predict))
    
    if bestModelScore < balancedScore:
        bestModelScore = balancedScore
        bestModel = "KFold Logistic Regression"

#Stratified Shuffle Split
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.7)
X = betterTrainingSet['question_text']
Y = betterTrainingSet['target']
for train_index, test_index in sss.split(X, Y):
    xTrain, xTest = X.iloc[train_index], X.iloc[test_index]
    yTrain, yTest = Y.iloc[train_index], Y.iloc[test_index]

    xTrain = bagOfWords.transform(xTrain)
    xTest = bagOfWords.transform(xTest)
    model = LogisticRegression().fit(xTrain, yTrain)
    score1 = model.score(xTestUnbalanced, yTestUnbalanced)
    score = model.score(xTest, yTest)
    predict = model.predict(xTestUnbalanced)
    balancedScore = metrics.balanced_accuracy_score(yTestUnbalanced, predict)

    print("Stratified Shuffle Split: Unbalanced Test:", score1, "Balanced Test:", score, "f1:", metrics.f1_score(yTestUnbalanced, predict), "f1 macro:", metrics.f1_score(yTestUnbalanced, predict, average="macro"), "f1 micro:", metrics.f1_score(yTestUnbalanced, predict, average="micro"), "f1 weighted:", metrics.f1_score(yTestUnbalanced, predict, average="weighted"), "f1 pos:", metrics.f1_score(yTestUnbalanced, predict, pos_label=0, average="binary"), "f1 neg:", metrics.f1_score(yTestUnbalanced, predict, pos_label=1, average="binary"), "precision:",
          metrics.precision_score(yTestUnbalanced, predict), "recall:", metrics.recall_score(yTestUnbalanced, predict), "balanced score:", balancedScore, "confusion matrix:", metrics.confusion_matrix(yTestUnbalanced, predict))
    
    if bestModelScore < balancedScore:
        bestModelScore = balancedScore
        bestModel = "Stratified Shuffle Split Logistic Regression"

#SVM
training_data, test_data = train_test_split(
    betterTrainingSet, train_size=0.1)
xTrain = bagOfWords.transform(training_data['question_text'])
xTest = bagOfWords.transform(test_data['question_text'])
yTrain = training_data['target']
yTest = test_data['target']

svclassifier = SVC(kernel='linear')
svclassifier.fit(xTrain, yTrain)

score = svclassifier.score(xTest, yTest)
score1 = svclassifier.score(xTestUnbalanced, yTestUnbalanced)
predict = svclassifier.predict(xTestUnbalanced)
balancedScore = metrics.balanced_accuracy_score(yTestUnbalanced, predict)

print("SVM: Balanced Test:", score, "f1:", metrics.f1_score(yTestUnbalanced, predict), "f1 macro:", metrics.f1_score(yTestUnbalanced, predict, average="macro"), "f1 micro:", metrics.f1_score(yTestUnbalanced, predict, average="micro"), "f1 weighted:", metrics.f1_score(yTestUnbalanced, predict, average="weighted"), "f1 pos:", metrics.f1_score(yTestUnbalanced, predict, pos_label=0, average="binary"), "f1 neg:", metrics.f1_score(yTestUnbalanced, predict, pos_label=1, average="binary"), "precision:",
      metrics.precision_score(yTestUnbalanced, predict), "recall:", metrics.recall_score(yTestUnbalanced, predict), "balanced score:", balancedScore, "confusion matrix:", metrics.confusion_matrix(yTestUnbalanced, predict))

if bestModelScore < balancedScore:
    bestModelScore = balancedScore
    bestModel = "SVM"


#TF-IDF Logistic Regression Model
tfidf_trfm = text.TfidfTransformer(norm=None)
xTrain = bagOfWords.transform(betterTrainingSet['question_text'])
yTrain = betterTrainingSet['target']
xTrainTfidf = tfidf_trfm.fit_transform(xTrain)
xTestTfidf = tfidf_trfm.transform(xTestUnbalanced)

tfidfModel = LogisticRegression().fit(xTrainTfidf, yTrain)
score = tfidfModel.score(xTestTfidf, yTestUnbalanced)
predict = tfidfModel.predict(xTestTfidf)
balancedScore = metrics.balanced_accuracy_score(yTestUnbalanced, predict)

print("TF-IDF: score:", score, "f1:", metrics.f1_score(yTestUnbalanced, predict), "f1 macro:", metrics.f1_score(yTestUnbalanced, predict, average="macro"), "f1 micro:", metrics.f1_score(yTestUnbalanced, predict, average="micro"), "f1 weighted:", metrics.f1_score(yTestUnbalanced, predict, average="weighted"), "f1 pos:", metrics.f1_score(yTestUnbalanced, predict, pos_label=0, average="binary"), "f1 neg:", metrics.f1_score(yTestUnbalanced, predict, pos_label=1, average="binary"), "precision:", metrics.precision_score(yTestUnbalanced, predict), "recall:", metrics.recall_score(
    yTestUnbalanced, predict), "balanced score:", balancedScore, "confusion matrix:", metrics.confusion_matrix(yTestUnbalanced, predict))

if bestModelScore < balancedScore:
    bestModelScore = balancedScore
    bestModel = "TF-IDF"


print("best model:", bestModel, "best model score:", bestModelScore)


#testdf = pd.read_csv('test.csv', usecols=['question_text'])
#xTestOfficial = bagOfWords.transform(testdf['question_text'])

#ynew = model.predict(xTestOfficial)
#for i in range(len(ynew)):
#    print("X:%s, Predicted=")
