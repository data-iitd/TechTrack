<!-- # EMNLP 2018 Update
Data and code related to our recent [EMNLP'18 paper] (https://arxiv.org/abs/1808.10012) is released on 31st Oct 2018.

**Code contributors: Bhavana Dalvi Mishra, Niket Tandon, Joel Grus

Detailed instructions to train your own ProStruct model can be found in: EMNLP18-README.md

To evaluate your model's predictions on the ProPara task (EMNLP'18),
please Download the evaluator code from a separate leaderboard repository: https://github.com/allenai/aristo-leaderboard/tree/master/propara


ProPara leaderboard is now live at: https://leaderboard.allenai.org/propara -->


Source Code:
The base code for this repository has been taken from https://github.com/allenai/propara

# TechTrack
The TechTrack dataset is track properties of diverse set of entities in technical procedural documents. <!-- For more details, read the [Thesis paper](SaranshGoyal.pdf).To have a better understanding of the model and the dataset, go through the Thesis paper. --> To understand the format of the dataset, refer to the ProPara dataset:
```
    Reasoning about Actions and State Changes by Injecting Commonsense Knowledge, Niket Tandon, Bhavana Dalvi Mishra, Joel Grus, Wen-tau Yih, Antoine Bosselut, Peter Clark, EMNLP 2018
```

These models are built using the PyTorch-based deep-learning NLP library, [AllenNLP](http://allennlp.org/).

 * ProLocal: A simple local model that takes a sentence and entity as input and predicts state changes happening to the entity.
 * Bert: A Bert-based classifier that takes as input a natural query and step text and builds a linear classifier on top of CLS embedding.

ProLocal and Bert are described in our paper.

<!--   ```
    Reasoning about Actions and State Changes by Injecting Commonsense Knowledge, Bhavana Dalvi Mishra, Lifu Huang, Niket Tandon, Wen-tau Yih, Peter Clark, NAACL 2018
  ```
  ** Bhavana Dalvi Mishra and Lifu Huang contributed equally to this work.


ProStruct model is described in our EMNLP'18 paper:
   ```
    Reasoning about Actions and State Changes by Injecting Commonsense Knowledge, Niket Tandon, Bhavana Dalvi Mishra, Joel Grus, Wen-tau Yih, Antoine Bosselut, Peter Clark, EMNLP 2018
   ```
   ** Niket Tandon and Bhavana Dalvi Mishra contributed equally to this work. -->

# Setup Instruction

1. Create the `propara` environment using Anaconda

  ```
  conda create -n propara python=3.7
  ```

2. Activate the environment

  ```
  source activate propara
  ```

3. Install the requirements in the environment: 

  ```
  pip install -r requirements.txt
  ```

<!-- 4. Test installation

 ```
 pytest -v -->
 <!-- ``` -->

<!-- # Download the dataset
You can download the ProPara dataset from
  ```
   http://data.allenai.org/propara/
  ```  -->

# Train your own models
Detailed instructions are given in the following READMEs:
 * [ProLocal](data/naacl18/prolocal/README.md)
 * [Bert](data/naacl18/bert/README.md)

# Scripts to use
Use various scripts in the root folder to run training and testing of various models and datasets
 * run_all.sh : train Bert model on all properties (including combined model)
 * run_all_comp.sh : train all models with all-False rows included
 * run_all_test.sh : train isOpened model and test on all folders
 * run_bert_test.sh : train combined Bert model and test on all properties
 * run_prolocal_all.sh : train ProLocal model for all properties

# Data Processing Scripts
Use the scripts from "data processing scripts" folder to parse data from wikihow pages and parse Brat output files to usable dataset formats
* State change type dataset for ProLocal model
* Natural Query, step and change type dataset for Bert model

The raw subfolder also contains some un-documented, intermediate and raw scripts which need not be used but are present in case needed.
