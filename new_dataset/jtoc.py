import json 
import csv

data=None 
with open('./valid.json','r',encoding='utf-8') as e: 
    data=json.load(e)   
write=open('valid.csv','w',encoding='utf-8') 
csv_write=csv.writer(write)
csv_write.writerow(data[0].keys())
for row in data: 
    csv_write.writerow(row.values()) 
write.close()
