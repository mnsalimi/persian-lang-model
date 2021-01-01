from random import shuffle
import csv
import re

datas = []
with open('../input/lobkalam/entry.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',',)
    for row in csv_reader:
        try:
            text = row[7].strip()
            if len(text)>20 and (text.find("کووید")>=0 or text.find("ویروس چینی")>=0 or text.find("کرونا")>=0):
                datas.append(row[7].strip())
        except:
            pass
datas = datas[:1000]
shuffle(datas)

total_words = 0
f = open("../output/prepared_data.txt", encoding="utf-8", mode = "w")
for i in datas:
    words = [item for item in re.split('; | |\.|\n|\.|\!|\?|\؟', i) if item!=""]
    total_words += len(words)
    f.write(i + "\n")

print(total_words)