import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import cupy as cp
import pandas as pd
import argparse
#from tqdm import tqdm


## Requirements for PPS() input:
# real_data: contains complete real dataset (train+test) [pandas DataFrame]
# train_data: dataset that was used by synthesizer to generate the synthetic dataset [pandas DataFrame]; part of real_data
# syn_data: synthetic data [pandas DataFrame]
# cat_feat: names of categorical features [List(str)]
# num_feat: names of numerical features [List(str)]

class PPS:
    def __init__(self, real_data, train_data, syn_data, cat_feat, num_feat, class_var):
        self.real_data = real_data
        self.train_data = train_data
        self.syn_data = syn_data
        self.cat_feat = cat_feat + [class_var]
        self.num_feat = num_feat
        self.class_var = class_var

        # Preprocessed datasets
        self.real_data_processed = None
        self.train_data_processed = None
        self.syn_data_processed = None

        # Results
        self.matches = []
        self.precision = 0
        self.recall = 0
        self.pps = 0

    # Preprocessing: Min-Max normalization for numerical features
    def preprocess(self, dataset):
        if len(self.num_feat) > 0:
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset[self.num_feat] = scaler.fit_transform(dataset[self.num_feat])
        return dataset

    # Preprocess all datasets
    def preprocess_datasets(self):
        self.real_data_processed = self.preprocess(self.real_data.copy())
        self.train_data_processed = self.preprocess(self.train_data.copy())
        self.syn_data_processed = self.preprocess(self.syn_data.copy())


    # compute nearest neighbor similarity
    def compute_similarity(self, r, dataset, cat_sim):

        if len(cat_sim.shape) > 2:
            print("incorrect cat_sim")
        r[self.cat_feat] = 1
        dataset[self.cat_feat] = cat_sim
        similarities = cosine_similarity(dataset.values, r.values.reshape(1, -1))[:,0]

        return np.sort(similarities)[-2:], similarities # if r = syn_r => nn_similarity -> [-1] else -> [-2]

    # Find matching records based on threshold similarity
    def find_matches(self, s, dataset, threshold):
        cat_sim = (s[self.cat_feat] == dataset[self.cat_feat]).astype(int).values
        similarities = self.compute_similarity(s, dataset.copy(), cat_sim)[1]
        return dataset[similarities >= threshold]


    def find_unique_match(self, train_set, matches):
        disclosure_risks_all = []
        train_set = cp.asarray(train_set.values)
        for _, train_matches, d in matches:
            match_found = cp.any(cp.all(cp.asarray(train_matches.values)[:, None, :] == train_set[None, :, :], axis=2), axis=0)
            disclosure_risks = cp.where(match_found, d, 0)
            disclosure_risks_all.append(disclosure_risks)

        disclosure_risks_all = cp.vstack(disclosure_risks_all)
        non_zero_cols = cp.any(disclosure_risks_all != 0, axis=0)
        disclosure_risks_all = disclosure_risks_all[:, non_zero_cols]
        high_risks = cp.amax(disclosure_risks_all, axis=0)
        return cp.asnumpy(high_risks)


    # Step 3: Perform membership inference analysis
    def run_analysis(self):

        # Step 1: Preprocess datasets and compute global threshold
        self.preprocess_datasets()

        # Compute global similarity threshold for real data
        cat_sim = (np.array(self.real_data_processed[self.cat_feat].values[:, np.newaxis]) == np.array(self.real_data_processed[self.cat_feat].values[np.newaxis, :])).astype(int)  # compares each row with all other rows => matrix: x=rows, y=rows, cell=comparison_val
        cat_sim = cp.asnumpy(cat_sim)
        global_similarities = self.real_data_processed.apply(
            lambda r: self.compute_similarity(r.copy(),self.real_data_processed.copy(),cat_sim[self.real_data_processed.index.get_loc(r.name)])[0][0], axis=1)
        global_threshold = np.mean(global_similarities)

        # Analyze each synthetic record
        #for _, s in tqdm(self.syn_data_processed.iterrows(),total=self.real_data_processed.shape[0], leave=True, desc="Over syn"):
        for _, s in self.syn_data_processed.iterrows():

            # Find real matches within the threshold
            real_matches = self.find_matches(s.copy(), self.real_data_processed.copy(), global_threshold)
            if real_matches.shape[0] == 0:
                continue

            # calculate local threshold for synthetic record (s)
            cat_sim = (cp.array(real_matches[self.cat_feat].values[:, np.newaxis]) == cp.array(real_matches[self.cat_feat].values[np.newaxis, :])).astype(int)  # compares each row with all other rows => matrix: x=rows, y=rows, cell=comparison_val
            cat_sim = cp.asnumpy(cat_sim)
            local_similarities = real_matches.apply(
                lambda r: self.compute_similarity(r.copy(), real_matches.copy(), cat_sim[real_matches.index.get_loc(r.name)])[0][0], axis=1)
            local_threshold = np.mean(local_similarities)

            # Step 2: Find training matches with the local threshold
            train_matches = self.find_matches(s.copy(), self.train_data_processed.copy(), local_threshold)

            # Disclosure risk: 0 if no training match, otherwise 1 / number of matches
            if train_matches.shape[0] == 0:
                continue

            disclosure_risk = 1 / train_matches.shape[0]

            # Append the matches
            self.matches.append((s, train_matches, disclosure_risk))

        # Step 3: privacy protection score
        if len(self.matches) == 0:
            self.pps = 1
            return self.pps

        # compute precision
        self.precision = np.mean(np.array(list(map(lambda x: x[2], self.matches)))) # average over all disclosure risks in matches

        # compute recall based on unique matches
        unique_matches = self.find_unique_match(self.train_data_processed.copy(), self.matches.copy())
        self.recall = np.sum(unique_matches) / self.train_data.shape[0]

        # compute privacy protection score (pps)
        self.pps = 1 - (self.precision + self.recall) / 2
        return self.pps


# start of command line call and loading of arguments
if __name__ == "__main__":

    # defines parsers
    parser = argparse.ArgumentParser()

    # arguments for data generation
    parser.add_argument('--train_path', type=str, required=True, help='path to train dataset (CSV) that was used by synthesizer to generate the synthetic dataset')
    parser.add_argument('--gen_path', type=str, required=True, help='path to synthetic dataset (CSV)')
    parser.add_argument('--test_path', type=str, required=True, help='path to test dataset (CSV) that was NOT used for synthesis')
    parser.add_argument('--cat_feat', type=str, required=False, nargs='+', default=[], help='list of categorical feature names')
    parser.add_argument('--num_feat', type=str, required=False, nargs='+', default=[], help='list of numerical feature names')
    parser.add_argument('--class_var', type=str, required=False, help='class name of datasets')
    args = parser.parse_args()

    # load datasets
    train_set = pd.read_csv(args.train_path, index_col=0)
    gen_data = pd.read_csv(args.gen_path, index_col=0)
    test_set = pd.read_csv(args.test_path, index_col=0)
    real_set = pd.concat([train_set, test_set], axis=0)  # combines train and test back to full real dataset needed for analysis

    # make sure all datasets share the same column type
    ct = train_set.columns.dtype
    sets = [train_set, test_set, real_set, gen_data]
    for s in sets:
        s.columns = s.columns.astype(ct)

    # assign categorical/numerical features names
    cat_feat_names = args.cat_feat
    num_feat_names = args.num_feat
    class_var = args.class_var

    # run privacy evaluation -> obtain pps
    pps_obj = PPS(real_set, train_set, gen_data, cat_feat_names, num_feat_names, class_var)
    pps = pps_obj.run_analysis()
    print("privacy protection score:", pps)

