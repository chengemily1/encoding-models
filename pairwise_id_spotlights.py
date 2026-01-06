#### Dependencies ####
import json
import numpy as np
import logging
import sys
import time
import joblib
import matplotlib.pyplot as plt
import torch
import cortex # This dependency is pycortex, which enables the plotting of flatmaps. It can be disabled.
from cvxopt import matrix, solvers # Only necessary for the stacked model.
from transformers import AutoTokenizer, AutoModelForCausalLM # Only necessary for feature extraction.
from dadapy import Data
import tqdm
import argparse 
from skdim.id import TwoNN

np.random.seed(40)

# Repository imports
from ridge_utils.ridge import bootstrap_ridge
import ridge_utils.npp
from ridge_utils.util import make_delayed
from ridge_utils.dsutils import make_word_ds
from ridge_utils.DataSequence import DataSequence
from ridge_utils.tokenization_helpers import generate_efficient_feat_dicts_opt
from ridge_utils.tokenization_helpers import convert_to_feature_mats_opt

### Some extra helper functions
zscore = lambda v: (v - v.mean(0)) / v.std(0)
zscore.__doc__ = """Z-scores (standardizes) each column of [v]."""
zs = zscore

## Matrix corr -- find correlation between each column of c1 and the corresponding column of c2
mcorr = lambda c1, c2: (zs(c1) * zs(c2)).mean(0)
mcorr.__doc__ = """Matrix correlation. Find the correlation between each column of [c1] and the corresponding column of [c2]."""

# Some parameters
NUM_VOX = 95556 # Number of voxels in the subject we plan to use
NUM_TRS = 790 # Number of TRs across 3 test stories
trim_start = 50 # Trim 50 TRs off the start of the story
trim_end = 5 # Trim 5 off the back
ndelays = 4 # We use 4 FIR delays (2 seconds, 4 seconds, 6 seconds, 8 seconds)
delays = range(1, ndelays + 1)

parser = argparse.ArgumentParser(prog='pairwise ID comp')
parser.add_argument('subject', type=str, choices=['UTS02', 'UTS03'])

# Get response data
resp_dict = joblib.load(f"{args.subject}_responses.jbl") # Located in story_responses folder

# Load in words 
with open('grids_cheap.txt', 'r') as f: 
    grids = [title.strip() for title in f.readlines()]

test_stories = ["wheretheressmoke", 'fromboyhoodtofatherhood', 'onapproachtopluto']
train_stories = [story for story in resp_dict.keys() if story in grids and story not in test_stories]

# Make voxel responses data
voxel_responses = np.vstack([resp_dict[story] for story in train_stories + test_stories])
np.random.shuffle(voxel_responses) # Shuffle along word dimension

# Compute the pairwise spotlight ID
dims_union = []
nseeds = 5
batchsize = int(voxel_responses.shape[0] // nseeds) + 1

# Precompute masks for all spotlights
data_spotlights = joblib.load(f"all_spotlights_{args.subject}_all_cortex.jbl")
spotlight_masks = [(s[-1] > 0) for s in data_spotlights]

# This can be optimized/made concurrent!
for j in tqdm.tqdm(range(len(data_spotlights)-1), desc='computing ID for spotlight unions'):
    mask1 = spotlight_masks[j]
    this_spotlight_dims = []

    for i in range(j+1, len(data_spotlights)):
        mask2 = spotlight_masks[i]

        # union mask
        overall_mask = mask1 | mask2

        seeds = []
        for k in range(0, len(voxel_responses), batchsize):
            batch = voxel_responses[k: k + batchsize, overall_mask]
            dada_data = Data(coordinates=batch, n_jobs=20)
            dada_data.compute_distances(maxk=5)
            
            # estimate ID
            id_twoNN, _, r = dada_data.compute_id_2NN()
            seeds.append(id_twoNN)

        this_spotlight_dims.append(seeds)

    dims_union.append(this_spotlight_dims)

# Now compute average IDs for each spotlight pair
avg_ids_union = []
for spotlight_pair_ids in dims_union:
    avg_ids_union.append([np.nanmean(ids) for ids in spotlight_pair_ids])

# Compute per-voxel ids_to_plot arrays (to plot on the flatmap)
ids_to_plot_list = []
for j, spotlight_pair_ids in enumerate(avg_ids_union):
    voxels_to_plot = np.zeros(voxel_responses.shape[-1])
    num_times_represented = np.zeros(voxel_responses.shape[-1])

    mask1 = spotlight_masks[j]
    for i, id_val in enumerate(spotlight_pair_ids):
        mask2 = spotlight_masks[j+1+i]
        union_mask = mask1 | mask2
        voxels_to_plot += union_mask * id_val
        num_times_represented += union_mask

    ids_to_plot_pair = voxels_to_plot / np.maximum(num_times_represented, 1)  # avoid div by zero
    ids_to_plot_list.append(ids_to_plot_pair.tolist())

# Save results for subject
RESULTS = {}
RESULTS['spotlight_ids'] = avg_ids_union.tolist()
RESULTS['ids_to_plot_list'] = ids_to_plot_list

with open(f'{args.subject}_pairwise.json', 'w') as f:
    json.dumps(RESULTS, f)


