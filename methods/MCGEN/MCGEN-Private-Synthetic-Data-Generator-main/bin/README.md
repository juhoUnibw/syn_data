# MC-GEN: Multi-level Clustering for Private Synthetic Data Generation
This repository contains the demo code of paper 'MC-GEN: Multi-level Clustering for Private Synthetic Data Generation' (https://arxiv.org/abs/2205.14298). 

## Features
- Generate private synthetic datasets

## Instruction
Import the *MC-GEN* into Eclipse as a maven project run the demo:

**Step 1**
Split your dataset using *DataSetSplit.java*. It will split the dataset by label and generate the seed datasets for next step.

**Step 2**
Generate the synthetic dataset using *generativeModel.java*.

**Step 3**
Test the synthetic dataset on classification tasks.
