import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')
import faulthandler
faulthandler.enable()
import sys
sys.path.append('methods/TabuLa/Tabula-main')
from tabula import Tabula # change tabula to tabula_middle_padding to test middle padding method


# TabuLa synthesis
def gen(trainSet, disc_feat_names, smpl_frac):

    # gpt2-medium is the larger version used in the original paper with 355M par (https://huggingface.co/openai-community/gpt2-medium)
    # distil-gpt2 has 82M pars (https://huggingface.co/distilbert/distilgpt2)
    # compression on trainSet not automatically?

    model = Tabula(llm='distilgpt2', experiment_dir = "IEEE", batch_size=4, epochs=200, categorical_columns = disc_feat_names) # 100/400 epochs for the smallest models in original paper. 200 chosen because same amount for GReaT.
    model.model.load_state_dict(torch.load("pretrained-model/model.pt", map_location=torch.device('cuda')), strict=False) # pre-trained model from paper (trained on Intrusion dataset). map_location=torch.device('cpu') only if no GPU available
    model.fit(trainSet)
    gen_data = model.sample(n_samples=int(trainSet.shape[0]*smpl_frac))

    return gen_data



