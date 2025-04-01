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


# Repository imports
from ridge_utils.ridge import bootstrap_ridge
import ridge_utils.npp
from ridge_utils.util import make_delayed
from ridge_utils.dsutils import make_word_ds
from ridge_utils.DataSequence import DataSequence
from ridge_utils.tokenization_helpers import generate_efficient_feat_dicts_opt
from ridge_utils.tokenization_helpers import convert_to_feature_mats_opt

from manifold_utils.projection import UpProjection, train_model
from manifold_utils.algorithms import *
from manifold_utils.constants import *

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
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--n_evecs", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--autoencoder_epochs", type=int, default=100)
    parser.add_argument("--autoencoder_lr", type=float, default=1e-3)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    # These files are located in the story_data folder of the Box
    grids = joblib.load("grids_huge.jbl") # Load TextGrids containing story annotations
    trfiles = joblib.load("trfiles_huge.jbl") # Load TRFiles containing TR information

    # We'll build an encoding model using this set of stories for this tutorial.
    train_stories = ['adollshouse', 'adventuresinsayingyes', 'alternateithicatom', 'avatar', 'buck', 'exorcism',
                'eyespy', 'fromboyhoodtofatherhood', 'hangtime', 'haveyoumethimyet', 'howtodraw', 'inamoment',
                'itsabox', 'legacy', 'naked', 'odetostepfather', 'sloth',
                'souls', 'stagefright', 'swimmingwithastronauts', 'thatthingonmyarm', 'theclosetthatateeverything',
                'tildeath', 'undertheinfluence']

    test_stories = ["wheretheressmoke"]

    # Filter out the other stories for the tutorial
    for story in list(grids):
        if story not in (train_stories + test_stories):
            del grids[story]
            del trfiles[story]

    # Make datasequence for story
    wordseqs = make_word_ds(grids, trfiles)

    # We will be using a sliding context window with minimum size 256 words that increases until size 512 words.
    tokenizer = AutoTokenizer.from_pretrained(args.model) # Same tokenizer for all sizes

    # Make dictionary to align tokens and words
    text_dict, text_dict2, text_dict3 = generate_efficient_feat_dicts_opt(wordseqs, tokenizer, 256, 512)

    # We are going to use the 125m parameter model for this tutorial, but any size should work 
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # We will extract features from the 9th layer of the model
    LAYER_NUM = args.layer

    start_time = time.time()
    for phrase in text_dict2:
        if text_dict2[phrase]:
            inputs = {}
            inputs['input_ids'] = torch.tensor([text_dict[phrase]]).int()
            inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape)
            out = list(model(**inputs, output_hidden_states=True)[2])
            out = out[LAYER_NUM][0].cpu().detach().numpy()
            out = np.array(out)
            this_key = tuple(inputs['input_ids'][0].cpu().detach().numpy())
            acc_true = 0
            for ei, i in enumerate(this_key):
                if this_key[:ei+1] in text_dict3:
                    acc_true += 1
                    text_dict3[this_key[:ei+1]] = out[ei,:]
    end_time = time.time()

    print("Feature extraction took", end_time - start_time, "seconds on", model.device)

    # Convert back from dictionary to matrix
    feats = convert_to_feature_mats_opt(wordseqs, tokenizer, 256, 512, text_dict3)

    #Training data
    Rstim = np.nan_to_num(np.vstack([ridge_utils.npp.zs(feats[story][10:-5]) for story in train_stories]))

    #Test data
    Pstim = np.nan_to_num(np.vstack([ridge_utils.npp.zs(feats[story][trim_start:-trim_end]) for story in test_stories]))

    # Add FIR delays
    delRstim = make_delayed(Rstim, delays)
    delPstim = make_delayed(Pstim, delays)

    # Get response data
    resp_dict = joblib.load("UTS03_responses.jbl") # Located in story_responses folder
    Rresp = np.vstack([resp_dict[story] for story in train_stories])
    Presp = np.vstack([resp_dict[story][40:] for story in test_stories])

    # Implement diffusion map
    # step 1: project the stim to low dimensions
    # print('Projecting x')
        # t = time.time()
    # dmap = dm.DiffusionMap.from_sklearn(n_evecs = 40, alpha = 0.5, epsilon = 'bgh', k=64) # choosing 40 eigenfunctions bc the ID ~ 40
    # delRstim_projected = dmap.fit_transform(delRstim)
    Mx_train = delRstim

        # step 2: project the response to low dimensions
    print('Projecting y')

    t = time.time()
    dmap_y = dm.DiffusionMap.from_sklearn(n_evecs = 250, alpha = 0.5, epsilon = 'bgh', k=300)
    Rresp_projected = dmap_y.fit_transform(Rresp)
    My_train = Rresp_projected
    np.save('My_train.npy', My_train)
    print(f'Ran in {time.time() - t}s')

    # eval R^2
    linear_model = Ridge(alpha=0.01)
    linear_model.fit(Mx_train, My_train)
    My_train_hat = linear_model.predict(Mx_train)

    print('Evaluating R2 in M space')
    R2_M = linear_model.score(Mx_train, My_train)

    print('Correlation: ', np.mean([pearsonr(My_train_hat[i], My_train[i])[0] for i in range(len(My_train_hat))]))
    print('R2 in fitted space: ', R2_M)

    # Make sure we can project back up to the original space.
    # step 4: learn a map from the projected response back up to the original
    input_dim = My_train.shape[-1]
    output_dim = Rresp.shape[-1]  

    My_train = torch.Tensor(My_train).to('cuda')
    Rresp = torch.Tensor(Rresp).to('cuda')
    train_dataset = TensorDataset(My_train, Rresp)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    My_test = dmap_y.transform(Presp)
    My_test = torch.Tensor(My_test).to('cuda')
    Presp = torch.Tensor(Presp).to('cuda')
    test_dataset = TensorDataset(My_test, Presp)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = UpProjection(input_dim, output_dim)
    train_model(model, train_dataloader, test_dataloader, epochs=args.autoencoder_epochs, lr=args.autoencoder_lr)

    # step 5: evaluate the R^2 in the original space
    My_train_hat = torch.Tensor(linear_model.predict(delRstim)).to('cuda')
    eval_dataset = TensorDataset(My_train_hat, torch.Tensor(Rresp).to('cuda'))
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    preds = []

    model.eval()
    for x_batch, _ in eval_dataloader:
        with torch.no_grad():
            pred = model(x_batch).detach().cpu().numpy()
            preds.append(pred)

    preds = np.vstack(preds)
    Rresp_cpu = Rresp.cpu().numpy()
    print(r2_score(preds, Rresp_cpu))
    print(np.corrcoef(preds, Rresp_cpu))

    # Test
    # step 1: project test stim 
    print('Projecting x')
    X_test = delPstim

    # step 2: project test response
    print('Projecting y')
    Presp_projected = dmap_y.transform(Presp)

    # step 3: apply linear map to reduced X, eval R^2 in reduced space
    My_test_hat = linear_model.predict(X_test)
    print('Test metrics in M space')
    print('R2 in reduced space: ', r2_score(My_test_hat, Presp_projected))
    print('Correlation in reduced space: ', np.mean([pearsonr(My_test_hat[i], Presp_projected[i])[0] for i in range(len(My_test_hat))]))

    # step 4: apply inv. map to M and eval R^2 in og space
    Presp_hat = model(My_test_hat)
    Presp_hat = Presp_hat.cpu().numpy()
    print('Test metrics in original space')
    print('Test R2: ', r2_score(Presp_hat, Presp))
    print('Test Correlation: ', np.mean([pearsonr(Presp_hat[i], Presp[i])[0] for i in range(len(Presp_hat))]))