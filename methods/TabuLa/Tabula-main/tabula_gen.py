import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')
import faulthandler
faulthandler.enable()
import sys
sys.path.append('methods/TabuLa/Tabula-main')
from tabula import Tabula # change tabula to tabula_middle_padding to test middle padding method
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

# TabuLa synthesis
def gen(trainSet, disc_feat_names, smpl_frac, class_var):

    # gpt2-medium is the larger version used in the original paper with 355M par (https://huggingface.co/openai-community/gpt2-medium)
    # distil-gpt2 has 82M pars (https://huggingface.co/distilbert/distilgpt2)
    # compression on trainSet not automatically?

    model = Tabula(llm='distilgpt2', experiment_dir = "methods/TabuLa/Tabula-main/chkpt", batch_size=8, epochs=100, categorical_columns = disc_feat_names+[str(class_var)]) # 100/400 epochs for the smallest models in original paper. 200 chosen because same amount for GReaT.
    model.model.load_state_dict(torch.load("methods/TabuLa/Tabula-main/pretrained-model/model.pt", map_location=torch.device('cuda')), strict=False) # pre-trained model from paper (trained on Intrusion dataset). map_location=torch.device('cpu') only if no GPU available
    trainSet.columns = trainSet.columns.astype(str)
    model.fit(trainSet, conditional_col=str(class_var))
    #model_path = "methods/TabuLa/saved_models"
    #torch.save(model.model.state_dict(), model_path + "/model.pt")
    #model.save(model_path)
    # model = model.load_from_dir(model_path)
    # model = Tabula.load_from_dir(model_path)
    gen_data = model.sample(n_samples=int(trainSet.shape[0]*smpl_frac), max_length=1000)

    return gen_data



