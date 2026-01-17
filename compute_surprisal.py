#### Dependencies ####
import numpy as np
from surprisal import AutoHuggingFaceModel
import logging
import sys
import time
import joblib
import argparse
import os
from sklearn.metrics import r2_score
import pdb
from tqdm import tqdm
import json
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("hi")


fpath = '/home/echeng/encoding-models/pile_sane_ds.txt'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)

    with open(fpath, 'r') as f:
        data = [dat.strip('\n').split('\t')[0] for dat in f.readlines()][:10000]

    # Load in model
    m = AutoHuggingFaceModel.from_pretrained(args.model, model_class='gpt')
    
    # Add bos token if it doesn't exist
    if m.tokenizer.bos_token_id is None:
        m.tokenizer.bos_token_id = m.tokenizer.pad_token_id
    
    # Compute surprisals
    def process_sentence(sentence):
        result = m.surprise(sentence, use_bos_token=True) # get a conditional dist for the first token as well. 
    
        return float(np.mean([float(s) for s in result[0].surprisals]))
    
    results = [process_sentence(sentence) for sentence in tqdm(data)]
    model_str = args.model.split('/')[-1]

    # Save
    save_dir = f'/home/echeng/encoding-models/results/{model_str}'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/surprisal.pkl', 'wb') as f:
        pickle.dump(results, f)
