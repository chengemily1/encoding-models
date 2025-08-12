#### Dependencies ####
import numpy as np
import logging
import sys
import time
import joblib
import torch
from cvxopt import matrix, solvers # Only necessary for the stacked model.
from transformers import AutoTokenizer, AutoModelForCausalLM # Only necessary for feature extraction.
from pydiffmap import diffusion_map as dm
import time
import argparse
import os
from sklearn.metrics import r2_score
import pdb
from tqdm import tqdm
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("hi")

# Repository imports
from ridge_utils.ridge import bootstrap_ridge_with_y_projection
import ridge_utils.npp
from ridge_utils.util import make_delayed
from ridge_utils.dsutils import make_word_ds

from manifold_utils.projection import down_project, get_up_projection_map, get_up_projections_torch
from manifold_utils.algorithms import *
from manifold_utils.constants import *
from manifold_utils.feature_extraction import FeatureExtractor

def test(x_project, linear, inv_map):
    """
        x_project: function that projects x to low dimensions
        y_project: function that projects y to low dimensions
        linear: linear map from projected x to projected y
        inv_map: inverse map from projected y to original y
    """
    return lambda x: inv_map(linear(x_project(x)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--y_projection", type=str, default="pca", choices=['pca', 'dm', 'I']) # I is identity projection
    parser.add_argument('--which_layers', type=str, default='single', help='feature selection algo', choices=['single', 'all', 'every_other', 'idCorr'])
    parser.add_argument("--n_layers", type=int, default=1, help="How many layers we want to include from the model")
    parser.add_argument("--seed_layer", type=int, default=9, help="the first layer to include (only layer is n_layers=1)") 
    parser.add_argument("--n_evecs", type=float, default=1000)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--autoencoder_epochs", type=int, default=1000)
    parser.add_argument("--autoencoder_lr", type=float, default=1e-3)

    args = parser.parse_args()

    if args.which_layers == 'all' or args.which_layers == 'every_other':
        args.n_layers = 0

    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)

    # These files are located in the story_data folder of the Box
    grids = joblib.load("grids_huge.jbl") # Load TextGrids containing story annotations
    trfiles = joblib.load("trfiles_huge.jbl") # Load TRFiles containing TR information
    resp_dict = joblib.load("UTS03_responses.jbl") # Located in story_responses folder

    # We'll build an encoding model using this set of stories for this tutorial.
    test_stories = ["wheretheressmoke", 'fromboyhoodtofatherhood', 'onapproachtopluto']
    train_stories = [story for story in resp_dict.keys() if story in grids and story not in test_stories]

    # Filter out the other stories for the tutorial
    for story in list(grids):
        if story not in (train_stories + test_stories):
            del grids[story]
            del trfiles[story]

    # Make datasequence for story
    wordseqs = make_word_ds(grids, trfiles)

    # We will extract features now
    feature_extractor = FeatureExtractor(wordseqs, args.model, train_stories, test_stories)

    # Convert back from dictionary to matrix
    print('getting features')
    os.environ["TOKENIZERS_PARALLELISM"] = "true" # multiprocessing with tokenizer in feature_extraction
    feats = feature_extractor.get_features(args.which_layers, seed_layer=args.seed_layer) # N stories x L layers x d (previously N stories x d)
    print('got the features')
    
    # Training data
    Rstim = np.nan_to_num(np.vstack([ridge_utils.npp.zs(feats[story][10:-5]) for story in train_stories]))

    # Test data
    Pstim = np.nan_to_num(np.vstack([ridge_utils.npp.zs(feats[story][trim_start:-trim_end]) for story in test_stories]))

    # Add FIR delays
    print('Adding FIR delays...')
    delRstim = make_delayed(Rstim, delays)
    delPstim = make_delayed(Pstim, delays)

    # Get response data
    Rresp = np.vstack([resp_dict[story] for story in train_stories]) # training Y
    Presp = np.vstack([resp_dict[story][40:] for story in test_stories]) # testing Y
    My_train = Rresp 
    My_test = Presp

    # Get explanatory variables
    Mx_train = delRstim
    Mx_test = delPstim
    
    # Bootstrap parameters
    alphas = np.logspace(1, 4, 1) # 15) # Equally log-spaced ridge parameters between 10 and 10000. 
    nboots = 1 #3 # Number of cross-validation ridge regression runs. You can lower this number to increase speed.
    chunklen = 20
    nchunks = int(len(Rresp) * 0.25 / chunklen)

    print('Computing projection maps on train data')
    _, projection_map_y = down_project(Rresp, project_type=args.y_projection, n_evecs=args.n_evecs)
    up_projection_map_y = get_up_projection_map(args, My_train, Rresp, Presp, project_type=args.y_projection, projection_map_y=projection_map_y)
    print("Bootstrap ridge")

    # Use RJ's bootstrap ridge code modified to handle projection
    wt, corr, best_alpha, bootstrap_corrs, valinds = bootstrap_ridge_with_y_projection(
                                                        Mx_train, My_train, Mx_test, My_test,
                                                        alphas, nboots, chunklen, nchunks,
                                                        up_projection_map_y, projection_map_y,
                                                        y_projection=args.y_projection,
                                                    )
    print("check the data type of the outputs")

    bootstrap_corrs = bootstrap_corrs.squeeze() # 1 x nvox x 1
    pdb.set_trace()
    results = {
        'params': vars(args),
        'corr': list(corr), # nvox
        'bscorrs': list(bootstrap_corrs),
        'val_indices': list(valinds), 
        'alphas': list(best_alpha) # nvox
    }
    model_str = args.model.split('/')[-1]

    # Save
    save_dir = f'/home/echeng/encoding-models/results/{model_str}'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/results_{args.which_layers}_n_layers_{args.n_layers}_seed_layer_{args.seed_layer}_y_rank_{args.n_evecs}_{args.y_projection}_ridge.json', 'w') as f:
        json.dump(results, f)
