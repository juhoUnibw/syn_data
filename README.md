# Utility Meets Privacy: A Critical Evaluation of Tabular Data Synthesizers
### Supplementary code for paper publication

This repository contains an evaluation pipeline for 12 public synthesizers, described in **Utility Meets Privacy: A Critical Evaluation of Tabular Data Synthesizers**. 
It may be used to reproduce the results of the study, or to evaluate further synthesizers. A membership inference analysis to generate privacy protection score 
for synthetic datasets (eval > pps.py) can be used independently; it is described below.


## Run Pipeline
Recommended is using the docker image with pre-installed libraries, but you may use the git repository as well if you wish. 

***Download the image***
LINK GIGAMOVE

***Load the image***
sudo docker load -i univnet.tar

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

'--data': Select datasets to derive synthetic data from (names are listed here). Default='all'.\
'--n_spl': Choose number of real data splits to use for synthesis. Default=10.\
'--method': Select synthesizers to generate data: tvae, gausscop, ctgan, arf, pzflow, knnmtd, mcgen, corgan, smote, priv_bayes, cart, great. Default='all' -> time-consuming, not recommended for test purposes.\
'--smpl_frac': Define synthetic data size (value is multiplied with training data size). Default=1.\
'--splits': Set to True if splits from paper should be used. Default=False.\
'--test': True for test purposes -> reduces data size if datasets>5000. Default=False.\

> 📌 **Note:** Depending on your resources, running the full pipeline (all synthesizers and datasets may result in memory overload), and take a long time.

Example:
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

'--data': Select datasets to evaluate for each synthesizer (names are listed here). Default='all'.\
'--n_spl': Choose number of data splits to evaluate (max as many synthetic datasets available per dataset-synthesizer combination). Default=10.\
'--method': Select synthesizer to evaluate: tvae, gausscop, ctgan, arf, pzflow, knnmtd, mcgen, corgan, smote, priv_bayes, cart, great. Default='all' -> time-consuming, not recommended for test purposes.\
'--weights': Define weighting of utility/privacy for utility-privacy-score calculation. Default=(0.5, 0.5).\
'--calc': Calculate evaluation scores. Default=False.\
'--summary': Generate summary statistics of evaluation results. Default=False.\

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

## Evaluate Privacy using the Privacy Protection Score (pps)

```bash
docker run -it ieee_gpu bash
```
```bash
python3 pps.py real_data_path train_data_path test_data_path gen_data_path
```