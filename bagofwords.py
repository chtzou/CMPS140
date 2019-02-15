#code borrowed from: https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/ch04.html
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

#current problems: there are less bad questions than good ones, so the model is unbalanced. need to balance the model.

df = pd.read_csv('train.csv', usecols=['question_text', 'target'])
goodQ = df[df.target == 0]
badQ = df[df.target == 1]
goodQ2 = goodQ.sample(len(badQ))

print(len(badQ)/(len(goodQ)+len(badQ)))


betterTrainingSet = pd.concat([badQ, goodQ2])


training_data_base, test_data_base = train_test_split(df, train_size=0.7, random_state=123)
training_data, test_data = train_test_split(betterTrainingSet, train_size=0.7,random_state=123)

bagOfWords = text.CountVectorizer()
xTrain = bagOfWords.fit_transform(training_data['question_text'])
xTest = bagOfWords.transform(test_data['question_text'])
xTrain2 = bagOfWords.fit_transform(training_data_base['question_text'])
xTest2 = bagOfWords.fit_transform(test_data_base['question_text'])

yTrain = training_data['target']
yTest = test_data['target']
yTrain2 = training_data_base['target']
yTest2 = test_data_base['target']


clf = DummyClassifier(strategy='most_frequent', random_state=0)
clf.fit(xTrain2, yTrain2)
print(clf.score(xTest2, yTest2))

model = LogisticRegression().fit(xTrain, yTrain)
score = model.score(xTest, yTest)

print(score)

#testdf = pd.read_csv('test.csv', usecols=['question_text'])
#xTestOfficial = bagOfWords.transform(testdf['question_text'])

#ynew = model.predict(xTestOfficial)
#for i in range(len(ynew)):
#    print("X:%s, Predicted=")
