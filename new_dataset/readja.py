import json
imagename=[]
classkind=[]
with open('./valid.json', 'r') as f:
    temp = json.loads(f.read())
    print(temp)
    print(len(temp))
