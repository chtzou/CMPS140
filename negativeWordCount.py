import csv
import re

filename='badQuestions.csv'

f = open(filename)
csv_f = csv.reader(f)

wordArray = []

for row in csv_f:
    line = row[0]
    line = re.findall(r"[\w']+|[.,!?;]", line)
    for i in line:
        if i.lower() not in wordArray:
            wordArray.append(i.lower())

w = open('wordCount.csv', 'w')

f.seek(0,0)

bagOfWords = []
lineIdx = 0
for row in csv_f:
    line = row[0]
    line = re.split(r"\W+", line)
    bagOfWords.append([0]*len(wordArray))
    for i in line:
        bagOfWords[lineIdx][wordArray.index(i.lower())]+=1
    lineIdx += 1

writer = csv.writer(w)
writer.writerow(wordArray)
print("here")
for row in bagOfWords:
    writer.writerow(row)
