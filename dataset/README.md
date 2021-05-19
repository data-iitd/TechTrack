# TechTrack Dataset

<!-- MarkdownTOC -->

1. [Directory Structure](#directory-structure)
	1. [Setup 1: Training each property Individually](#setup-1-training-each-property-individually)
	1. [Setup 2: Training the Combined Model](#setup-2-training-the-combined-model)
1. [How to use it?](#how-to-use-it)

<!-- /MarkdownTOC -->


<a id="directory-structure"></a>
## Directory Structure

<a id="setup-1-training-each-property-individually"></a>
### Setup 1: Training each property Individually
In our first setup, we trained the models for each property individually. 

[`./setup_1/bert`](setup_1/bert/) contains the data for BERT model and [`./setup_1/prolocal`](setup_1/prolocal/) for ProLocal model.

<a id="setup-2-training-the-combined-model"></a>
### Setup 2: Training the Combined Model
Along with individual models for each property, we also train a combined BERT-based classifier to predict property values for each property type. Here, we train a single model on multiple properties, with each property trained on respective training rows. 

[`./setup_2/bert`](setup_2/bert/) contains the data for this setup.


<a id="how-to-use-it"></a>
## How to use it?
All the files for the desired setup has to be placed in [`data/Inputs`](../data/Inputs). This can be done manually or use the scripts provided in the root. For more info, see the [root REAMDE](../README.md).
