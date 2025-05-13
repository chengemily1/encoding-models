import typing
import torch
import time 
from tqdm import tqdm 
import pdb 
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM # Only necessary for feature extraction.

from ridge_utils.tokenization_helpers import generate_efficient_feat_dicts_opt, convert_to_feature_mats_opt

class FeatureExtractor:
    """
        This class takes as input a model and text inputs,
        then selects the relevant features.
    """
    def __init__(self, wordseqs, model_str: str):
        # Model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_str) # Same tokenizer for all sizes
        self.model = AutoModelForCausalLM.from_pretrained(model_str, device_map='auto')

        self.wordseqs = wordseqs

        # Input text format and model 
        self.text_dict, self.text_dict2, self.text_dict3 = generate_efficient_feat_dicts_opt(wordseqs, self.tokenizer, 256, 512)

        # Features stored in self.text_dict3
        self.text_dict3 = self._extract_features()

        # Memory management
        del self.model 


    def _extract_features(self):
        start_time = time.time()
        print('Extracting features')
        for phrase in tqdm(self.text_dict2):
            if self.text_dict2[phrase]:
                inputs = {}
                inputs['input_ids'] = torch.tensor([self.text_dict[phrase]]).int().to(self.model.device)
                inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape).to(self.model.device)
                out = torch.cat(self.model(**inputs, output_hidden_states=True)[2], dim=0) # L layers x N_toks x hidden_dim
                out = out.cpu().detach().numpy()
                out = np.array(out)

                this_key = tuple(inputs['input_ids'][0].cpu().detach().numpy())
                acc_true = 0
                for ei, _ in enumerate(this_key):
                    if this_key[:ei+1] in self.text_dict3:
                        acc_true += 1
                        self.text_dict3[this_key[:ei+1]] = out[:, ei, :] # index into the correct token
        end_time = time.time()
        print("Feature extraction took", end_time - start_time, "seconds on", self.model.device)

        return self.text_dict3
    

    def get_features(self, selection_method: str, seed_layer = None):
        """

        Args:
            selection_method (str): selection_method in "single layer", "idCorr"

        Returns:
            np.array : feature matrix N x d
        """
        # result is N x L layers x d dimensions
        result = convert_to_feature_mats_opt(self.wordseqs, self.tokenizer, 256, 512, self.text_dict3)
        N, L, d = result.shape

        # memory management
        del self.tokenizer

        # Select features
        if selection_method == 'single layer':
            return result[:,seed_layer,:].squeeze(1)
        elif selection_method == 'all':
            return np.reshape(result, (N, L * d))
        elif selection_method == 'idCorr':
            # TODO needs to be implemented
            pass 
        