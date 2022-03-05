import random
import pickle
from collections import namedtuple 
from sklearn.model_selection import train_test_split
import numpy as np

Transition = namedtuple('Transition',['entity', 'property', 'value'])

with open('data/tuples_v2_with_add_info.pkl', 'rb') as f:
    d = pickle.load(f)
    data = d['data']
    topics_map = d['topics_map']

X, y = [], []

for k, v in data.items():
    y.append(topics_map[k])
    X.append(v)

# Get train, val, test split in ratio 0.75:0.1:0.15 of data_v
X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y, shuffle=True)
X_train, X_val = train_test_split(X_train, test_size=0.1, random_state=42, stratify=y_train, shuffle=True)

count = {}
for x in X:
    for step in x:
        count[len(step)] = count.get(len(step), 0) + 1
        
    


print("Total number of steps:", sum(count.values()))
print(">1 steps:", sum([v for k, v in count.items() if k > 1]))

arr = [0]*(max(count.keys())+1)
for k, v in count.items():
    arr[k] = v

for i, v in enumerate(arr):
    print(i, v, sep=' ')