# Utility Meets Privacy: A Critical Evaluation of Tabular Data Synthesizers
### Supplementary code for paper publication

This repository contains the evaluation pipeline described in the paper **Utility Meets Privacy: A Critical Evaluation of Tabular Data Synthesizers**. 
It can be used to reproduce the experimental part of the study, and may be adapted for evaluation of other synthesizers. The pipeline includes a script for privacy analysis of synthetic datasets (**pps.py**), which may also be used independently as described [here](#privacy-analysis-using-the-privacy-protection-score).


## Run Pipeline
For replication of the papers' experiments, use of a docker image with pre-installed libraries is recommended.
It contains the 12 synthesizers and 17 datasets described in the paper. Download and use of the docker image are described below. 
For evaluation of further synthesizers or datasets, feel free to adapt the code provided in the repository. 

### List of available datasets and synthesizers

Datasets:\
**breast_cancer, breast_tissue, cardiotocography, kidney, dermatology, diabetes,
retinography, echocardiogram, heart, lymphography, patient, stroke, thoracic_surgery,
thyroid, tumor, sani, eye**

Synthesizers:\
**tvae, gausscop, ctgan, arf, pzflow, knnmtd, mcgen, corgan, smote, priv_bayes, cart, great**


### Prerequisites

The following dependencies are required.

1. CUDA 12.7 and GPU
2. Docker 27.4.1 (see https://docs.docker.com/engine/install/)
3. Nvidia Container Toolkit 1.17.3 with Docker configuration (see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
5. Docker image > [Download](https://gigamove.rwth-aachen.de/de/download/b61eb655cc82ed32f69a66c5e2e1ebf9) [pw: ` fexwyb-qogfAs-7warty `]
available until 2025-04-28

>📌 **Note: if permission is denied when running docker, try the following:** 
> - use **sudo** prefix
> - relocate docker root directory (see https://www.ibm.com/docs/en/z-logdata-analytics/5.1.0?topic=software-relocating-docker-root-directory)

### Load docker image
```
docker load -i ieee_gpu.tar
```

>📌 **Note: if you want to create a new image after code adaptions, you may use the Dockerfile in this repository**
>: ```docker build -t [image_name] .```

### Generate synthetic data

```bash
docker run \
-v [abs_path]/gen_data:/app/eval/gen_data -w /app --gpus "device=0" ieee_gpu \
python3 IEEE_pipeline.py gen \
--data [dataset names] \
--n_spl [int] \
--method [synthesizer_names] \
--smpl_frac [float] \
--splits True \
--test True
```

**'--data'**: select [datasets](#list-of-available-datasets-and-synthesizers) to derive synthetic data from. Default='all'.\
**'--n_spl'**: choose number of real data splits to use for synthesis. Default=10.\
**'--method'**: select one or more [synthesizers](#list-of-available-datasets-and-synthesizers). Default='all' is time-consuming.\
**'--smpl_frac'**: define synthetic data size (value is multiplied with training data size). Default=1.\
**'--splits'**: set to True if data splits (train/test) from paper should be used. Default=False.\
**'--test'**: True for test purposes -> limits dataset sizes to max 5000. Default=False.\

> 📌 **Note:** Rather select subsets of synthesizers and datasets. Depending on your resources, running the full pipeline is time-consuming and may result in memory overload.  

***Example:***
```bash
docker run \
-v /home/julian_hoellig/ieee/gen_data:/app/eval/gen_data -w /app --gpus '"device=0"' ieee_gpu \
python3 IEEE_pipeline.py gen \
--n_spl 3 \
--method arf mcgen smote cart \
--data breast_cancer breast_tissue \
--test True
```

### Evaluate synthetic data

```bash
docker run \
[abs_path]/gen_data:/app/eval/gen_data [abs_path]/results:/app/eval/results -w /app --gpus "device=0" ieee_gpu \
python3 IEEE_pipeline.py eval \
--real_train_path /app/eval/train_data \
--gen_data_path /app/eval/gen_data \
--real_test_path /app/eval/test_data \
--n_spl [int] \
--calc True \
--summary True \
--method [synthesizer_names] \
--data [dataset_names] \
--weights ([w_us],[w_pps])
```
**-v**: mounts local directories into the docker container. Make sure the **gen_data** directory contains the synthetic datasets generated from previous synthesis.\
**'--data'**: select [datasets](#list-of-available-datasets-and-synthesizers) to evaluate for each synthesizer. Default='all'.\
**'--n_spl'**: choose number of data splits to evaluate (max as many synthetic datasets available per dataset-synthesizer combination). Default=10.\
**'--method'**: select [synthesizers](#list-of-available-datasets-and-synthesizers) included in evaluation. Default='all' -> time-consuming.\
**'--weights'**: define weighting of utility/privacy for utility-privacy-score calculation. Default=(0.5, 0.5).\
**'--calc'**: calculate evaluation results. Default=False.\
**'--summary'**: generate summary statistics of evaluation results. Default=False.\

> 📌 **Note:** Consider evaluating subsets of synthesizers and datasets. Depending on your resources, running the full pipeline is time-consuming and may result in memory overload.

**Results**\
The summary results can be found in the mounted **results** directory. They contain scores for the selected synthesizers, averaged across the selected datasets.
Scores on the individual datasets can be found in the mounted **gen_data** directory.

***Example***
```bash
docker run \
-v /home/julian_hoellig/ieee/gen_data:/app/eval/gen_data -v /home/julian_hoellig/ieee/results:/app/eval/results \
-w /app --gpus '"device=0"' ieee_gpu python3 IEEE_pipeline.py eval \
--real_train_path /app/eval/train_data \
--gen_data_path /app/eval/gen_data \
--real_test_path /app/eval/test_data \
--n_spl 3 \
--calc True \
--summary True \
--method mcgen cart smote arf \
--data breast_cancer breast_tissue
```

## Privacy Analysis using the privacy protection score

### Prerequisites

1. python 3.11
2. NumPy
3. pandas
4. CuPy (requires GPU access and CUDA)

> 📌 **Note:** CuPy speeds up the analysis considerably by using GPU capacities. 
> However, if you want to run on CPU, you can replace CuPy operations (prefixed with **cp.**) with NumPy operations.

### Run privacy analysis

```bash
python pps.py \
--train_path [path_to_train_data] \
--test_path [path_to_test_data] \
--gen_path [path_to_gen_data] \
--num_feat [list_of_numerical_feature_names] \
--class_var [list_of_categorical_feature_names]
```

**--train_path**: path to train dataset (CSV) that was used to generate the synthetic dataset.\
**--test_path**: path to test dataset (CSV) that was NOT used for synthesis.\
**--gen_path**: path to synthetic dataset (CSV)'.

> 📌 **Note:** 
> 1) Train and test datasets were split 50/50 from the same real dataset.
> 2) All feature names of the dataset must be specified in **--num_feat** and **--cat_feat**. 
> If all features are either numerical or categorical, only one of the arguments is required.
> 2) Train, test and synthetic datasets must be in CSV format and have the same features.

***Example***
```bash
python pps.py \
--train_path ex_data/train.csv \
--test_path ex_data/test.csv \
--gen_path ex_data/gen.csv \
--num_feat I0 PA500 HFS DA Area A/DA 'Max IP' DR P \
--class_var Class
```

### Citation

If you use the pps code for your research, please cite the following source:

J. Höllig and M. Geierhos, "Utility Meets Privacy: A Critical Evaluation of Tabular Data Synthesizers," in IEEE Access, doi: 10.1109/ACCESS.2025.3549680.
keywords: {Synthesizers;Synthetic data;Data privacy;Machine learning;Data models;Privacy;Deep learning;Accuracy;Protection;Predictive models;Membership inference analysis;privacy evaluation;tabular data synthesizer;utility-privacy trade-off},

BibTeX:
```bash
@ARTICLE{10918632,
  author={Höllig, Julian and Geierhos, Michaela},
  journal={IEEE Access}, 
  title={Utility Meets Privacy: A Critical Evaluation of Tabular Data Synthesizers}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Synthesizers;Synthetic data;Data privacy;Machine learning;Data models;Privacy;Deep learning;Accuracy;Protection;Predictive models;Membership inference analysis;privacy evaluation;tabular data synthesizer;utility-privacy trade-off},
  doi={10.1109/ACCESS.2025.3549680}}

```

## Licenses

This repository is licensed under MIT license. It contains or requires installation of the following third-party codes and repositories:
- arf: Adversarial Random Forests [10.32614/CRAN.package.arf] (GPL license).
- cor-gan [https://github.com/astorfi/cor-gan] (authors allow use with attribution - please cite their paper when using this repository in your study)
- be_great [https://github.com/kathrinse/be_great?tab=readme-ov-file] (MIT license)
- MC-GEN: Multi-level Clustering for Private Synthetic Data Generation [https://github.com/mingchenli/MCGEN-Private-Synthetic-Data-Generator?] (Apache license 2.0)
- pzflow [https://github.com/jfcrenshaw/pzflow] (MIT license)
- SMOTE [https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html#imblearn.over_sampling.SMOTE] (MIT license)
- dpart [https://pypi.org/project/dpart/] (MIT license)
- kNNMTD [https://github.com/jsivaku1/kNNMTD] (authors allow use with attribution - please cite their paper when using this repository in your study)
- SDV [https://github.com/sdv-dev/SDV] (Business Source License 1.1)

Attribution cor-gan:
A. Torfi and A. Fox, "CorGAN: Correlation-Capturing Convolutional Generative Adversarial Networks for Generating Synthetic Healthcare Records," 2020, arXiv:2001.09346.

Attribution kNNMTD:
J. Sivakumar, K. Ramamurthy, M. Radhakrishnan, and D. Won. "Synthetic sampling from small datasets: A modified mega-trend diffusion approach using k-nearest neighbors," Knowledge-Based Systems, 2021, p. 107687.
