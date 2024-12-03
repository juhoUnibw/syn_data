import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from toolz import unique
from tqdm import tqdm
tqdm.pandas()
import cupy as cp


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
            scaler = MinMaxScaler()
            dataset[self.num_feat] = scaler.fit_transform(dataset[self.num_feat])
        return dataset

    # Preprocess all datasets
    def preprocess_datasets(self):
        self.real_data_processed = self.preprocess(self.real_data.copy())
        self.train_data_processed = self.preprocess(self.train_data.copy())
        self.syn_data_processed = self.preprocess(self.syn_data.copy())

    # compute nearest neighbor similarity
    def compute_similarity(self, r, dataset, cat_sim):
        # other solutions: one-hot-encoding for cat features; Hamming distance between cat features, and cosine between numerical;
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
        return dataset[similarities >= threshold] # index or col?

    # Find unique match for training record
    def find_unique_match(self, t, matches):
        disclosure_risks_t = []
        for _, train_matches, d in tqdm(matches):
            disclosure_risks_t.append(d) if (train_matches == t).all(axis=1).any() else disclosure_risks_t.append(0) # append 0 if t not in train_matches (so that index of risks aligns with index of matches; important for next step)
        i = disclosure_risks_t.index(max(disclosure_risks_t)) # index of highest disclosure risk
        unique_match = matches[i]
        return unique_match

    # Step 3: Perform membership inference analysis
    def run_analysis(self):

        # Step 1: Preprocess datasets and compute global threshold
        self.preprocess_datasets()

        # Compute global similarity threshold for real data
        print("compute sim mat")
        cat_sim = (cp.array(self.real_data_processed[self.cat_feat].values[:, np.newaxis]) == cp.array(self.real_data_processed[self.cat_feat].values[np.newaxis, :]).astype(int))  # compares each row with all other rows => matrix: x=rows, y=rows, cell=comparison_val
        print("compute sim")
        global_similarities = self.real_data_processed.apply(
            lambda r: self.compute_similarity(r.copy(),self.real_data_processed.copy(),cat_sim[self.real_data_processed.index.get_loc(r.name)])[0][0], axis=1)
        global_threshold = np.mean(global_similarities)

        # Analyze each synthetic record
        #for _, s in tqdm(self.syn_data_processed.iterrows(),total=self.real_data_processed.shape[0], leave=True, desc="Over syn"):
        for _, s in tqdm(self.syn_data_processed.iterrows()):

            # Find real matches within the threshold
            real_matches = self.find_matches(s.copy(), self.real_data_processed.copy(), global_threshold)
            if real_matches.shape[0] == 0:
                continue

            # calculate local threshold for synthetic record (s)
            cat_sim = (real_matches[self.cat_feat].values[:, np.newaxis] == real_matches[self.cat_feat].values[np.newaxis, :]).astype(int)  # compares each row with all other rows => matrix: x=rows, y=rows, cell=comparison_val
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

        print("loop over")
        # compute precision
        print("pr")
        self.precision = np.mean(np.array(list(map(lambda x: x[2], self.matches)))) # average over all disclosure risks in matches

        # compute recall based on unique matches
        print("um")
        unique_matches = self.train_data_processed.apply(lambda t: self.find_unique_match(t.copy(), self.matches.copy()), axis=1)
        print("re")
        self.recall = np.sum(np.array(list(map(lambda x: x[2], unique_matches)))) / self.train_data.shape[0]

        print("pps")
        # Step 5: Compute Privacy Protection Score (PPS)
        self.pps = 1 - (self.precision + self.recall) / 2
        print("over")
        return self.pps

