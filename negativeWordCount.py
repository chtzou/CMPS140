import csv
import re

filename='badQuestions.csv'

f = open(filename)
csv_f = csv.reader(f)

ignore=['if', 'the', 'then', 'what', 'a', 'why', 'who', 'them', 'she', 'we', 'they', 'and', 'be', 'to', 'of', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'say', 'her', 'or', 'will', 'an', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'about', 'get', 'which', 'go', 'when', 'me', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'person', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', "it's", 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', ',', '.', '?', '!', ':']

wordCount = {}

for row in csv_f:
    line = row[0]
    line = re.findall(r"[\w']+|[.,!?;]", line)
    for i in line:
        if i.lower() not in wordCount and i.lower() not in ignore:
            wordCount[i.lower()] = 1
        elif i.lower() not in ignore:
            wordCount[i.lower()] += 1

w = open('wordCount.csv', 'w')

for key in wordCount:
    w.write("%s,%s\n"%(key,wordCount[key]))

print(wordCount)
