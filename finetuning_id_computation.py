#### Dependencies ####
import numpy as np
import logging
import sys
import time
import joblib
import torch
import time
import argparse
import os
from sklearn.metrics import r2_score
import pdb
from tqdm import tqdm
import json
import os
from dadapy import Data
from contextlib import contextmanager
from joblib import Parallel, delayed
from transformers import AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("hi")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, choices=['UTS02-finetuned', 'UTS03-finetuned', 'pretrained'])
    parser.add_argument('--modality', type=str, choices=['fmri', 'ecog'])

    args = parser.parse_args()

    return args

def compute_gride(reps):
    results = {layer: {'id': [],
                       'err': [],
                       'r': []
                       } for layer in reps}
    for layer, layer_reps in tqdm(reps.items(), desc='id over layers'):
        _data = Data(coordinates=layer_reps, n_jobs=10)
        _data.remove_identical_points()
    
        # estimate ID
        ids_scaling, ids_scaling_err, rs_scaling = _data.return_id_scaling_gride(range_max = 2**13)
        results[layer]['r'] = rs_scaling.tolist()
        results[layer]['err'] = ids_scaling_err.tolist()
        results[layer]['id'] = ids_scaling.tolist()

    return results

if __name__ == "__main__":
    args = parse_args()
    print(args)

    LAYERS = list(range(13))

    if args.modality == 'ecog':
        fpath_format = '/home/echeng/encoding-models/{}/layer.{}/podcast.npz'
        reps = {}
        
        for layer in LAYERS:
            path = fpath_format.format(args.experiment, layer)
            with np.load(path) as data:
                reps[layer] = data["features"]
                
    elif args.modality == 'fmri':
        if args.experiment == 'pretrained': 
            fpath = '/home/echeng/encoding-models/fmri/features_downsamp_pretrained.joblib'
        elif args.experiment == 'UTS02-finetuned':
            fpath = '/home/echeng/encoding-models/fmri/UTS02/features_epoch_9.joblib'
        elif args.experiment == 'UTS03-finetuned':
            fpath = '/home/echeng/encoding-models/fmri/UTS03/features_epoch_9.joblib'

        reps = joblib.load(fpath)
        reps = np.vstack([reps[story] for story in reps])
        reps = reps[np.random.permutation(reps.shape[0])] # N x d randomized.

        batches = np.array_split(reps, 5)
        for i, batch in enumerate(batches):
            
        

    RESULTS = {}
    
    print('Experiment ', args.experiment)
    RESULTS[args.experiment] = compute_gride(reps)
    
    with open(f'results/wavlm-base_gride_{args.experiment}.json', 'w') as f:
        json.dump(RESULTS, f)
    