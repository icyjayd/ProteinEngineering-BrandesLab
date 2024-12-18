import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate
from datetime import datetime
import time
import joblib
from tqdm import tqdm
import numpy as np
from pprint import pprint
import pandas as pd
import os

def one_hot_encode(seq, max_len=None):
    amino_acids = "".join(["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"])
    mapping = dict(zip(amino_acids, range(len(amino_acids))))
    seq2 = [mapping[i] for i in seq]
    seq2 = np.eye(len(amino_acids))[seq2]
    if max_len is not None:
        if len(seq2) < max_len:
            seq2 = np.vstack([seq2, np.zeros((max_len - len(seq2), len(amino_acids)))])
    return seq2

def one_hot_multiple(seqs, max_len=None, bar=False):
  if bar:
    out = []
    for seq in tqdm(seqs):
      out.append(one_hot_encode(seq, max_len))
    seqs = out
  else:
    seqs = [one_hot_encode(seq, max_len) for seq in seqs]
  seqs = np.array(seqs)
  seqs = seqs.reshape(seqs.shape[0], -1)
  return seqs

def array_id2idxs(cat_file):
    length = len(cat_file)
    if length <=2:
        cat = '0'
        file_idx = cat_file
    else:
        cat = cat_file[0]
        file_idx = cat_file[1:]
    return (cat, file_idx)

if __name__ == "__main__":
    #get and parse argument
    parser = argparse.ArgumentParser(description="Perform random forest on protein gym DMS substitutions scores per category with 1 CV split")
    parser.add_argument("cat_file", type=str)
    parser.add_argument("--model", type=str, default='rf')
    parser.add_argument('--limit', type=bool, default=False)
    parser.add_argument('--cv', type=int, default=5)
    args = parser.parse_args()
    cat_file = args.cat_file
    #split/cat/file = digits
    #e.g. 0407 = split 0, category 4, file 7
    assert len(cat_file) <=3, "only 5 categories and less than 99 filenames"
    #
    # assert array_id2idxs('0') == ('0', '0', '0')
    # assert array_id2idxs('10') == ('0', '0', '10')
    # assert array_id2idxs('207') == ('0', '2', '07')
    # assert array_id2idxs('1006') == ('1', '0', '06')
    # assert array_id2idxs('1070') == ('1', '0', '70')
    cat, file_idx = array_id2idxs(cat_file)
    model_type = args.model
    limit = args.limit
    cv = args.cv
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    if limit:
        limit = "_limit_1024"
    else:
        limit = ""
    #get date
    now = datetime.now()
    year = str(now.year)
    month = str(now.month)
    if len(month) <2:
        month = "0" + month
    day = str(now.day)
    if len(day) <2:
        day = "0" + day
    day = "-".join([year, month, day])

    #get files
    dir = '/gpfs/scratch/jic286/BrandesLab/DMS_subs_files'
    meta = pd.read_csv(os.path.join(dir, 'DMS_substitutions.csv'))
    cat = meta['coarse_selection_type'].unique()[int(cat)]
    start = time.time()
    file_prefix = os.path.join(dir, f'{cat}{limit}')

    #get indices
    df = pd.read_csv(f'{file_prefix}.csv')
    assert int(file_idx) < df['filename'].nunique(), "file_idx index exceeds number of files"
    filename = df['filename'].unique()[int(file_idx)]
    gene = filename[:filename.find('_')]
    print(f"Training {model} model on {cat} DMS scores for gene {gene} on {day} with 5f CV.")
    df = df[df['filename']==filename].reset_index(drop=True)
    print(f"Number of sequences: {len(df)}")
    max_len = max(df['mutated_sequence'].apply(lambda x: len(x)))

    X = one_hot_multiple(df['mutated_sequence'], max_len=max_len)
    y = df['DMS_score']

    end = time.time()
    scoring = ['neg_root_mean_squared_error', 'r2']
    scores = cross_validate(model, X, y, scoring=scoring, cv=cv,
    return_estimator=True, return_indices = True)
    print(f"Training took {round(end - start)} seconds", flush=True)

    out = f'/gpfs/scratch/jic286/BrandesLab/model_data/{gene}_DMS_sub_{cat}_{model_type}_{limit}_{cv}f_cv_{day}.pkl'
    joblib.dump(scores, out)
    print(f"Mean test neg RMSE: {scores['test_neg_root_mean_squared_error'].mean()}")
    print(f'Mean test R^2: {scores["test_r2"].mean()}')
