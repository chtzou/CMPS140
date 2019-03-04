#code borrowed from: https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/ch04.html
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn import metrics

#current problems: there are less bad questions than good ones, so the model is unbalanced. need to balance the model.

dfOG = pd.read_csv('train.csv', usecols=['question_text', 'target'])
df = dfOG[:len(dfOG)//2] 
df2 = dfOG[len(dfOG)//2:] #This is to split in half so we have two completely separate tests to test with
goodQ = df[df.target == 0]
badQ = df[df.target == 1]
goodQ2 = goodQ.sample(len(badQ))

betterTrainingSet = pd.concat([badQ, goodQ2])

training_data_base, test_data_base = train_test_split(df2, train_size=0.7, random_state=123)

#Base Model
bagOfWords = text.CountVectorizer()
xTrainUnbalanced = bagOfWords.fit_transform(training_data_base['question_text'])
xTestUnbalanced = bagOfWords.transform(test_data_base['question_text'])

yTrainUnbalanced = training_data_base['target']
yTestUnbalanced = test_data_base['target']


#clf = DummyClassifier(strategy='most_frequent', random_state=0)
#clf.fit(xTrainUnbalanced, yTrainUnbalanced)
#print("base model", clf.score(xTestUnbalanced, yTestUnbalanced))

#Logistic Regression Model
training_data, test_data = train_test_split(
    betterTrainingSet, train_size=0.7, random_state=123)
xTrain = bagOfWords.transform(training_data['question_text'])
xTest = bagOfWords.transform(test_data['question_text'])

yTrain = training_data['target']
yTest = test_data['target']

model = LogisticRegression().fit(xTrain, yTrain)
score1 = model.score(xTestUnbalanced, yTestUnbalanced)
score = model.score(xTest, yTest)
predict = model.predict(xTestUnbalanced)

print("Pure Logistic Regression: Unbalanced Test:", score1, "Balanced Test:", score, "f1:", metrics.f1_score(yTestUnbalanced, predict), "precision:",
      metrics.precision_score(yTestUnbalanced, predict), "recall:", metrics.recall_score(yTestUnbalanced, predict), "balanced score:", metrics.balanced_accuracy_score(yTestUnbalanced, predict), "confusion matrix:", metrics.confusion_matrix(yTestUnbalanced, predict))

#TF-IDF Model
#tfidf_trfm = text.TfidfTransformer(norm=None)
#xTrainTfidf = tfidf_trfm.fit_transform(xTrain)
#xTestTfidf = tfidf_trfm.transform(xTest)

#tfidfModel = LogisticRegression().fit(xTrainTfidf, yTrain)
#score = tfidfModel.score(xTestTfidf, yTest)

#print("TF-IDF", score)

#Logistic Regression-Balanced
balancedModel = LogisticRegression(class_weight='balanced').fit(xTrainUnbalanced, yTrainUnbalanced)
score1 = balancedModel.score(xTestUnbalanced, yTestUnbalanced)
score = balancedModel.score(xTest, yTest)
predict = balancedModel.predict(xTestUnbalanced)

print("Balanced: Unbalanced Test:", score1, "Balanced Test:", score, "f1:", metrics.f1_score(yTestUnbalanced, predict), "precision:", metrics.precision_score(yTestUnbalanced, predict), "recall:", metrics.recall_score(yTestUnbalanced, predict), "balanced score:", metrics.balanced_accuracy_score(yTestUnbalanced, predict), "confusion matrix:", metrics.confusion_matrix(yTestUnbalanced, predict))


#testdf = pd.read_csv('test.csv', usecols=['question_text'])
#xTestOfficial = bagOfWords.transform(testdf['question_text'])

#ynew = model.predict(xTestOfficial)
#for i in range(len(ynew)):
#    print("X:%s, Predicted=")
