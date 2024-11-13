import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')
from be_great import GReaT
import faulthandler
faulthandler.enable()

# GReaT synthesis
def gen(trainSet, smpl_frac, llm_id):

    # gpt2-medium is the larger version used in the original paper with 355M par (https://huggingface.co/openai-community/gpt2-medium)
    # distil-gpt2 has 82M pars (https://huggingface.co/distilbert/distilgpt2)
    device = 'cuda'
    model = GReaT(llm=llm_id, batch_size=4, epochs=200) # 200 epochs were trained in the original paper. Batch size is varied between 8-124, depending on the GPU size.
    trainSet.columns = trainSet.columns.astype(str)
    model.fit(trainSet)
    gen_data = model.sample(n_samples=int(trainSet.shape[0]*smpl_frac))

    return gen_data
