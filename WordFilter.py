import csv

infile = open('train.csv')
csv_file = csv.reader(infile)

bad_questions = []
for row in csv_file:
    if row[2] == '1':
        bad_questions.append(row[1])

    
outfile = open('badQuestions.csv', 'w')
write_csv = csv.writer(outfile)

for question in bad_questions:
    write_csv.writerow([question])