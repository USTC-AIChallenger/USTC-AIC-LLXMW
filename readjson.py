import json
imagename=[]
classkind=[]
with open('./eval73.json', 'r') as f:
    temp = json.loads(f.read())
    print(temp)
    print(len(temp))
