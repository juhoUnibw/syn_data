import warnings
warnings.filterwarnings('ignore')
from be_great import GReaT
import faulthandler
faulthandler.enable()

# GReaT synthesis
def gen(trainSet, smpl_frac, llm_id):

    # gpt2-medium is the larger version used in the original paper with 355M par (https://huggingface.co/openai-community/gpt2-medium)
    # distil-gpt2 has 82M pars (https://huggingface.co/distilbert/distilgpt2)
    model = GReaT(llm=llm_id, batch_size=8, epochs=200) # Batch size is varied between 8-124 in the original paper, depending on the GPU size.
    trainSet.columns = trainSet.columns.astype(str)
    model.fit(trainSet)
    model_path = "methods/GReaT/saved_models"
    model.save(model_path)
    #model = model.load_from_dir(model_path) # to load trained checkpoint
    #model = GReaT.load_from_dir(model_path)
    gen_data = model.sample(n_samples=int(trainSet.shape[0]*smpl_frac), max_length=1000)

    return gen_data
