import csv
import json
import os
ctoj=[]
with open('eval73.csv','r') as csv_f:
    csv_reader = csv.DictReader(csv_f)
    for row in csv_reader:
        d = {}
        for k, v in row.items():
            d[k] = v
            #print(row.items()
            
        print(d)
        print(d['disease_class'])
        d['disease_class']=int(d['disease_class'])
        print(d['disease_class'])
        ctoj.append(d)
with open('./eval73.json','w',encoding='utf-8') as json_file:
    json.dump(ctoj,json_file,ensure_ascii=False)
