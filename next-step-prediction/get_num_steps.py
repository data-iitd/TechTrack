import random
import pickle
from collections import namedtuple 
import numpy as np

Transition = namedtuple('Transition',['entity', 'property', 'value'])

with open('processed_data/traversed_data.pkl', 'rb') as f:
    d = pickle.load(f)['val']


count = {}
for x in d:
    count[len(x)] = count.get(len(x), 0) + 1

print(count)

arr = [0]*(max(count.keys())+1)
for k, v in count.items():
    arr[k] = v

for i, v in enumerate(arr):
    print(i, v, sep=' ')

'''
for i,v in enumerate(arr):
    if v==0: continue
    print(*([i]*v), sep=' ', end = ' ')
'''