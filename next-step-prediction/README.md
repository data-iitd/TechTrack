## Next step prediction

### Task
Given a sequence of performed transitions (“given sequence”), rank the transitions from the candidates set based on the transition that should follow, i.e., the most likely to follow must be ranked first.

### Performance
|                           	|   VAL  	|        	|  TEST  	|        	|
|---------------------------	|:------:	|:------:	|:------:	|:------:	|
|                           	|    MRR 	|    ACC 	|    MRR 	|    ACC 	|
| LSTM Baseline             	| 0.0180 	| 0.0056 	| 0.0205 	| 0.0103 	|
| Constrained LSTM Baseline 	| 0.0454 	| 0.0168 	| 0.0537 	| 0.0138 	|
