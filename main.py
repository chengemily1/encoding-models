#### Dependencies ####
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
from pydiffmap import diffusion_map as dm
from sklearn.linear_model import Ridge, LinearRegression
import time
import argparse
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import pdb
from scipy.stats import pearsonr
from tqdm import tqdm
import json

# Repository imports
from ridge_utils.ridge import bootstrap_ridge
import ridge_utils.npp
from ridge_utils.util import make_delayed
from ridge_utils.dsutils import make_word_ds
from ridge_utils.DataSequence import DataSequence
from ridge_utils.tokenization_helpers import generate_efficient_feat_dicts_opt, convert_to_feature_mats_opt, generate_efficient_feat_dicts_llama, convert_to_feature_mats_llama

from manifold_utils.projection import down_project, get_up_projection_map, get_up_projections_torch
from manifold_utils.algorithms import *
from manifold_utils.constants import *
from manifold_utils.utils import print_stats
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
    parser.add_argument('--which_layers', type=str, default='single', help='feature selection algo', choices=['single', 'all', 'idCorr'])
    parser.add_argument("--n_layers", type=int, default=1, help="How many layers we want to include from the model")
    parser.add_argument("--seed_layer", type=int, default=9, help="the first layer to include (only layer is n_layers=1)") 
    parser.add_argument("--n_evecs", type=float, default=1000)
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha in ridge regression")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--autoencoder_epochs", type=int, default=1000)
    parser.add_argument("--autoencoder_lr", type=float, default=1e-3)

    args = parser.parse_args()

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

    # We will be using a sliding context window with minimum size 256 words that increases until size 512 words.
    # tokenizer = AutoTokenizer.from_pretrained(args.model) # Same tokenizer for all sizes

    # Make dictionary to align tokens and words
    # TODO make compatible with tokenizers
    # text_dict, text_dict2, text_dict3 = generate_efficient_feat_dicts_opt(wordseqs, tokenizer, 256, 512)

    # Load the model
    # model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto')
    # print('Loaded model and tokenizer')

    # We will extract features now
    feature_extractor = FeatureExtractor(wordseqs, args.model)
    # LAYER_NUM = args.seed_layer

    # start_time = time.time()
    # print('Extracting features')
    # for phrase in tqdm(text_dict2):
    #     if text_dict2[phrase]:
    #         inputs = {}
    #         inputs['input_ids'] = torch.tensor([text_dict[phrase]]).int().to(model.device)
    #         inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape).to(model.device)
    #         out = list(model(**inputs, output_hidden_states=True)[2])
    #         out = out[LAYER_NUM][0].cpu().detach().numpy()
    #         out = np.array(out)

    #         this_key = tuple(inputs['input_ids'][0].cpu().detach().numpy())
    #         acc_true = 0
    #         for ei, i in enumerate(this_key):
    #             if this_key[:ei+1] in text_dict3:
    #                 acc_true += 1
    #                 text_dict3[this_key[:ei+1]] = out[ei,:]
    # end_time = time.time()

    # print("Feature extraction took", end_time - start_time, "seconds on", model.device)
    # del model # memory management

    # Convert back from dictionary to matrix
    feats = feature_extractor.get_features(args.which_layers, seed_layer=args.seed_layer) # N stories x L layers x d (previously N stories x d)

    # pdb.set_trace()
    # feats = convert_to_feature_mats_opt(wordseqs, tokenizer, 256, 512, text_dict3)
    # print('What is the dimension of feats')
    pdb.set_trace()

    #Training data
    Rstim = np.nan_to_num(np.vstack([ridge_utils.npp.zs(feats[story][10:-5]) for story in train_stories]))

    #Test data
    Pstim = np.nan_to_num(np.vstack([ridge_utils.npp.zs(feats[story][trim_start:-trim_end]) for story in test_stories]))

    # Add FIR delays
    delRstim = make_delayed(Rstim, delays)
    delPstim = make_delayed(Pstim, delays)

    # Get response data
    Rresp = np.vstack([resp_dict[story] for story in train_stories])
    Presp = np.vstack([resp_dict[story][40:] for story in test_stories])

    # Get explanatory variables
    Mx_train = delRstim
    Mx_test = delPstim

    # step 2: project the response to low dimensions
    print('Projecting y')
    t = time.time()

    My_train, projection_map_y = down_project(Rresp, project_type=args.y_projection, n_evecs=args.n_evecs)
    My_test = projection_map_y.transform(Presp)

    print(f'Ran in {time.time() - t:.2f}s')

    # eval R^2
    linear_model = Ridge(alpha=0.05)
    linear_model.fit(Mx_train, My_train)

    print('TRAIN metrics in M space=============================')
    My_train_hat = linear_model.predict(Mx_train)
    R2_M, correlations_M = print_stats(My_train_hat, My_train)

    print('TEST metrics in M space==============================')
    My_test_hat = linear_model.predict(Mx_test)
    R2_M_test, correlations_M_test = print_stats(My_test_hat, My_test)

    # Make sure we can project back up to the original space.
    # step 4: learn a map from the projected response back up to the original
    up_projection_map_y = get_up_projection_map(args, My_train, Rresp, Presp, project_type=args.y_projection, projection_map_y=projection_map_y)

    # step 5: evaluate the R^2 in the original space
    if args.y_projection == 'dm':
        My_train_hat = torch.Tensor(linear_model.predict(delRstim)).to('cuda')
        y_hat = get_up_projections_torch(My_train_hat, Rresp, up_projection_map_y)
    else:
        y_hat = up_projection_map_y(My_train_hat)

    print('TRAIN metrics in response space============================')
    R2_response, correlations_response = print_stats(y_hat, Rresp)

    print('TEST metrics in response space==============================')
    y_test_hat = up_projection_map_y(My_test_hat)
    R2_response_test, correlations_response_test = print_stats(y_test_hat, Presp)

    results = {
        'params': vars(args),
        'R2_M': float(R2_M),
        'R2_M_test': R2_M_test,
        'R2_response': R2_response,
        'R2_response_test': R2_response_test,
        'correlations_M': correlations_M,
        'correlations_M_test': correlations_M_test,
        'correlations_response': correlations_response,
        'correlations_response_test': correlations_response_test
    }
    model_str = args.model.split('/')[-1]

    # Save
    save_dir = f'/home/echeng/encoding-models/results/{model_str}'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/results_layer_{args.layer}_rank_{args.n_evecs}_{args.y_projection}.json', 'w') as f:
        json.dump(results, f)
