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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("hi")

# Repository imports
import ridge_utils.npp
from ridge_utils.util import make_delayed
from ridge_utils.dsutils import make_word_ds

from manifold_utils.algorithms import *
from manifold_utils.constants import *
from manifold_utils.feature_extraction import FeatureExtractor
from manifold_utils.utils import get_layer_order


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--subject', type=int, choices=[2,3])
    parser.add_argument('--add_delays', type=int, default=0)

    args = parser.parse_args()

    return args

def extract_llm_features(model, train_stories, test_stories, add_delay=False):
    if 'whisper' not in model and 'wavlm' not in model:
        grids = joblib.load("grids_huge.jbl") # Load TextGrids containing story annotations
        trfiles = joblib.load("trfiles_huge.jbl") # Load TRFiles containing TR information
        
        # Filter out the other stories for the tutorial
        for story in list(grids):
            if story not in train_stories + test_stories:
                del grids[story]
                del trfiles[story]

        # Make datasequence for story
        wordseqs = make_word_ds(grids, trfiles)

        # We will extract features now
        feature_extractor = FeatureExtractor(wordseqs, args.model, train_stories, test_stories)

        # Convert back from dictionary to matrix
        print('getting features')
        os.environ["TOKENIZERS_PARALLELISM"] = "true" # multiprocessing with tokenizer in feature_extraction
        feats = feature_extractor.get_features('all', seed_layer=0) # N stories x L layers x d (previously N stories x d)
        n_layers = feature_extractor.L_layers
        print('got the features')
    elif 'whisper' in args.model:
        feats = {}
        for seed_layer in range(0, 33, 2):
            features_path = f"/home/echeng/encoding-models/whisper-features/downsampled_featureseqs_whisper-large_layer{seed_layer}.jbl"
            feats_layer = joblib.load(features_path)  
            for story in feats_layer:
                if story not in feats: feats[story] = []
                feats[story].append(feats_layer[story])
        n_layers = len(feats[list(feats.keys())[0]])
        feats = {story: np.array(feats[story]).transpose(1, 0, 2) for story in feats}
        feats = {story: feats[story].reshape(feats[story].shape[0], feats[story].shape[1] * feats[story].shape[2]) for story in feats}

    elif 'wavlm' in args.model:
        n_layers = 25

        # Load directly from file
        features_path = '/home/echeng/encoding-models/wavlm-large_downsampled/layer.{}/{}.npz'        
        feats = { # story: N x (L x D)
            story: np.array([np.load(features_path.format(seed_layer, story))['features'] for seed_layer in range(n_layers)]).transpose(1, 0, 2) for story in tqdm(stories, desc='Loading features')
        }
        feats = {story: feats[story].reshape(feats[story].shape[0], feats[story].shape[1] * feats[story].shape[2]) for story in feats}

    print(f'model has {n_layers} layers')

    # Training data
    all_features = np.nan_to_num(np.vstack([ridge_utils.npp.zs(feats[story][10:-5]) for story in train_stories + test_stories]))
    
    # Add FIR delays
    if add_delay:
        print('Adding FIR delays...')
        all_features = make_delayed(all_features, delays)
        
    return all_features, n_layers

def get_ID(data):
    dada_data= Data(coordinates=data,n_jobs=10)
    dada_data.compute_distances(maxk = 5)
                
    # estimate ID
    id_twoNN, _, r = dada_data.compute_id_2NN()
    return id_twoNN

if __name__ == "__main__":
    args = parse_args()
    print(args)

    # Load spotlights
    resp_dict = joblib.load(f"UTS0{args.subject}_responses.jbl") # Located in story_responses folder
    data_spotlights = joblib.load(f"all_spotlights_UTS0{args.subject}_all_cortex.jbl")

    with open('grids_cheap.txt', 'r') as f: # avoid oom
        grids = [title.strip() for title in f.readlines()]

    # We'll build an encoding model using this set of stories for this tutorial.
    test_stories = ["wheretheressmoke", 'fromboyhoodtofatherhood', 'onapproachtopluto']
    train_stories = [story for story in resp_dict.keys() if story in grids and story not in test_stories]
    stories = train_stories + test_stories
    voxel_responses = np.vstack([resp_dict[story] for story in train_stories + test_stories]) # N sentences x V voxels

    # Step 1: load LLM features (all layers)
    LLM_features, n_layers = extract_llm_features(args.model, train_stories, test_stories, add_delay=args.add_delays)
    LLM_features = LLM_features.reshape(LLM_features.shape[0], n_layers, LLM_features.shape[-1] // n_layers) # N x L x d

    # Step 3: randomize the llm features and also the voxel responses (but in the same order)
    randomization = np.random.permutation(len(voxel_responses)) # break autocorrelation
    voxel_responses = voxel_responses[randomization]
    LLM_features = LLM_features[randomization,:,:]
    
    # Step 3: for layers x spotlights: compute ID(LLM), union ID = ID(concat(LLM, spotlight), ID spotlight.
    dims = []
    nseeds = 5
    batchsize = int(voxel_responses.shape[0] // nseeds) + 1        

    layer_ids = [None for _ in range(n_layers)]
    spotlight_ids = [None for _ in range(len(data_spotlights))]
    spotlight_layer_ids = [[None for _ in range(len(data_spotlights))] for _ in range(n_layers)] # layer x spotlight

    # 1. Precompute layer and spotlight batches
    llm_batches = [LLM_features[:batchsize, layer, :] for layer in range(n_layers)]
    
    voxel_batches = []
    for spotlight in tqdm(data_spotlights, desc='precomputing voxel batches'):
        mask = spotlight[-1] > 0
        voxel_batches.append(voxel_responses[:batchsize, mask])

    
    # For loop over the pairs
    for spotlight_i, spotlight in tqdm(enumerate(data_spotlights), desc='computing pairwise id for spotlights'): 
        voxel_batch = voxel_batches[spotlight_i]

        # spotlight-only id
        spotlight_ids[spotlight_i] = float(get_ID(voxel_batch))
        
        for layer in range(n_layers): 
            llm_batch = llm_batches[layer]
            union_batch = np.concatenate([voxel_batch, llm_batch], axis=1) 

            # layer-only ID
            if spotlight_i == 0: # compute the layer IDs on the first loop
                layer_ids[layer] = float(get_ID(llm_batch))

            # union ID
            spotlight_layer_ids[layer][spotlight_i] = float(get_ID(union_batch)) 
    
    results = {
        'spotlights': spotlight_ids, # n_spotlights
        'layers': layer_ids, # n_layers
        'union': spotlight_layer_ids, # n_layers x n_spotlights
    }
    model_str = args.model.split('/')[-1]

    # Save
    save_dir = f'/home/echeng/encoding-models/results/{model_str}/UTS0{args.subject}'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/id_summaries_spotlights_with_delay_{args.add_delays}.json', 'w') as f:
        json.dump(results, f)
