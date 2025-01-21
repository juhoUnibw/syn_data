import argparse
import os
import torch
import pandas as pd
import time
import random
import subprocess
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


subprocess.run("export KMP_DUPLICATE_LIB_OK=TRUE", shell=True, capture_output=True)
result = subprocess.run("echo $KMP_DUPLICATE_LIB_OK", shell=True, capture_output=True)

parser = argparse.ArgumentParser()

# experimentName is the current file name without extension
experimentName = os.path.splitext(os.path.basename(__file__))[0]

parser.add_argument("--DATASETPATH", type=str,
                    default=os.path.expanduser('~/data/MIMIC/processed/out_binary.matrix'),
                    help="Dataset file")

parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--n_epochs_pretrain", type=int, default=100,
                    help="number of epochs of pretraining the autoencoder")
parser.add_argument("--batch_size", type=int, default=512, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="l2 regularization")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument('--n_iter_D', type=int, default=5, help='number of D iters per each G iter')

# Check the details
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)

parser.add_argument("--cuda", type=bool, default=False, help="CUDA activation")
parser.add_argument("--multiplegpu", type=bool, default=True,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--num_gpu", type=int, default=2, help="Number of GPUs in case of multiple GPU")

parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between samples")
parser.add_argument("--epoch_time_show", type=bool, default=False, help="interval betwen image samples")
parser.add_argument("--epoch_save_model_freq", type=int, default=100, help="number of epops per model save")
parser.add_argument("--minibatch_averaging", type=bool, default=False, help="Minibatch averaging")

parser.add_argument("--training", type=bool, default=False, help="Training status")
parser.add_argument("--resume", type=bool, default=False, help="Training status")
parser.add_argument("--finetuning", type=bool, default=False, help="Training status")
parser.add_argument("--generate", type=bool, default=False, help="Generating Sythetic Data")
parser.add_argument("--class_var", help="class variable of dataset")
parser.add_argument("--cl", help="class information of dataset") # a model needs to be trained for each class data > was not implemented in original code
parser.add_argument("--evaluate", type=bool, default=False, help="Evaluation status")
parser.add_argument("--expPATH", type=str, default=os.path.expanduser('~/experiments/pytorch/model/'+experimentName),
                    help="Training status")
parser.add_argument("--smpl_frac", type=float, default=1, help="size of fake sample: fraction of original sample")
opt = parser.parse_args()

# Create experiments DIR
if not os.path.exists(opt.expPATH):
    os.system('mkdir {0}'.format(opt.expPATH))

# Random seed for pytorch
opt.manualSeed = random.randint(1, 10000) # fix seed
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
cudnn.benchmark = True

# Check cuda
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device BUT it is not in use...")

# Activate CUDA
device = torch.device("cuda" if opt.cuda else "cpu")

##########################
### Dataset Processing ###
##########################

data = pd.read_csv(opt.DATASETPATH, index_col=0)
feat = list(data.columns)
feat.remove(opt.class_var)
# last line is discrete feature information >> has to be extracted
disc_feat = data[feat].iloc[-1,:]
data.drop(index=data.iloc[-1,:].name, inplace=True)
if data[opt.class_var].dtype == 'int': # command line value is by default taken in string format
    opt.cl = int(opt.cl)
cl_data = data[feat][data[opt.class_var]==opt.cl]
data = np.asarray(cl_data)

sample_size = data.shape[0]
feature_size = data.shape[1]

# Split train-test
indices = np.random.permutation(sample_size)
training_idx, test_idx = indices[:int(0.8 * sample_size)], indices[int(0.8 * sample_size):]
if len(test_idx) < opt.batch_size:
    opt.batch_size = len(test_idx)
trainData = data[training_idx, :]
testData = data[test_idx, :]

# transform Object array to float
trainData = trainData.astype(np.float32)
testData = testData.astype(np.float32)

# ave synthetic data
np.save(os.path.join(opt.expPATH, "dataTrain.npy"), trainData, allow_pickle=False)
np.save(os.path.join(opt.expPATH, "dataTest.npy"), testData, allow_pickle=False)
class Dataset:
    def __init__(self, data, transform=None):

        # Transform
        self.transform = transform

        # load data
        self.data = data
        self.sample_size = data.shape[0]
        self.feature_size = data.shape[1]

    def return_data(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        #sample = np.clip(sample, 0, 1) # the authors decide to produce discrete binary features as input to the model >> not suitable for all datasets

        if self.transform:
           pass

        return torch.from_numpy(sample)


# Train data loader
dataset_train_object = Dataset(data=trainData, transform=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
dataloader_train = DataLoader(dataset_train_object, batch_size=opt.batch_size,
                              shuffle=False, num_workers=0, drop_last=True, sampler=samplerRandom)

# Test data loader
dataset_test_object = Dataset(data=testData, transform=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_test_object, replacement=True)
dataloader_test = DataLoader(dataset_test_object, batch_size=opt.batch_size,
                             shuffle=False, num_workers=0, drop_last=True, sampler=samplerRandom)


####################
### Architecture ###
####################

# to easily print inbetween layers of class Autoencoder
class Autoencoder2(nn.Module):

    def __init__(self):
        super(Autoencoder2, self).__init__()
        n_channels_base = 4

    def forward(self, x):
        print("between layers:",x.shape, x[0])
        return x
    
class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        n_channels_base = 4

        # ---- ADAPTION: with low feature sizes the conv filter size have to be adapted to avoid errors in the NN architecture (output size of one layer cant be below filter size of next layer) ----
        ## the default values are given before any adaption

        def adapt_filter_size(in_size, k_size, stride):
            if stride > 1:
                stride -= 1
            else:
                if k_size > 1:
                    k_size -= 1

            return k_size, stride
        
        def get_out_size(in_size, k_size, stride):
            #out_size = ((in_size - k_size) / stride) + 1
            out_size = ((in_size - (k_size-1) - 1) / stride) + 1 # formula: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

            return out_size

        def check_sizes(in_size, k_size, stride, k_size_next):
            out_size = get_out_size(in_size, k_size, stride)
            while out_size < k_size_next:
                if k_size == 1 and stride == 1:
                    k_size_next -= 1
                k_size, stride = adapt_filter_size(in_size, k_size, stride)
                out_size = get_out_size(in_size, k_size, stride)

            return out_size, k_size, stride, k_size_next

        # default sizes of original architecture
        in_size_1, k_size_1, stride_1 = feature_size, 5, 2
        k_size_2, stride_2 = 5, 2
        k_size_3, stride_3 = 5, 3
        k_size_4, stride_4 = 5, 3
        k_size_5, stride_5 = 5, 3
        k_size_6, stride_6 = 8, 1
        
        # conv 1
        #print("Conv1 before in={} k={} s={}:".format(in_size_1, k_size_1, stride_1))
        in_size_2, k_size_1, stride_1, k_size_2 = check_sizes(in_size_1, k_size_1, stride_1, k_size_2)
        #print("Conv1 after in={} k={} s={}:".format(in_size_1, k_size_1, stride_1))
        # conv 2
        #print("Conv2 before in={} k={} s={}:".format(in_size_2, k_size_2, stride_2))
        in_size_3, k_size_2, stride_2, k_size_3 = check_sizes(in_size_2, k_size_2, stride_2, k_size_3)
        #print("Conv2 after in={} k={} s={}:".format(in_size_2, k_size_2, stride_2))
        # conv 3
        #print("Conv3 before in={} k={} s={}:".format(in_size_3, k_size_3, stride_3))
        in_size_4, k_size_3, stride_3, k_size_4 = check_sizes(in_size_3, k_size_3, stride_3, k_size_4)
        #print("Conv3 after in={} k={} s={}:".format(in_size_3, k_size_3, stride_3))
        # conv  4
        #print("Conv4 before in={} k={} s={}:".format(in_size_4, k_size_4, stride_4))
        in_size_5, k_size_4, stride_4, k_size_5 = check_sizes(in_size_4, k_size_4, stride_4, k_size_5)
        #print("Conv4 after in={} k={} s={}:".format(in_size_4, k_size_4, stride_4))
        # conv 5
        #print("Conv5 before in={} k={} s={}:".format(in_size_5, k_size_5, stride_5))
        in_size_6, k_size_5, stride_5, k_size_6 = check_sizes(in_size_5, k_size_5, stride_5, k_size_6)
        #print("Conv5 after in={} k={} s={}:".format(in_size_5, k_size_5, stride_5))
        # conv 6
        #print("Conv6 before in={} k={} s={}:".format(in_size_6, k_size_6, stride_6))
        out_size_enc, k_size_6, stride_6, k_size_6 = check_sizes(in_size_6, k_size_6, stride_6, k_size_6)
        #print("Conv6 after in={} k={} s={}:".format(in_size_6, k_size_6, stride_6))

        # ---- ADAPATION END ---- 

        self.encoder = nn.Sequential(
            
           # Autoencoder2(),
            nn.Conv1d(in_channels=1, out_channels=n_channels_base, kernel_size=k_size_1, stride=stride_1, padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
           # Autoencoder2(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=n_channels_base, out_channels=2 * n_channels_base, kernel_size=k_size_2, stride=stride_2, padding=0,
                      dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
                #      Autoencoder2(),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=2 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=k_size_3, stride=stride_3,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
               #     Autoencoder2(),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=4 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=k_size_4, stride=stride_4,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
                 #     Autoencoder2(),
            nn.BatchNorm1d(8 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=8 * n_channels_base, out_channels=16 * n_channels_base, kernel_size=k_size_5, stride=stride_5,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
                  #    Autoencoder2(),
            nn.BatchNorm1d(16 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=16 * n_channels_base, out_channels=32 * n_channels_base, kernel_size=k_size_6, stride=stride_6,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
              #        Autoencoder2(),
            nn.Tanh(), # creates values between -1 and 1
        #    Autoencoder2(),
        )


        def adapt_filter_size(in_size, k_size, stride):
            if stride > 1:
                stride -= 1
            else:
                k_size -= 1

            return k_size, stride
        
        def get_out_size(in_size, k_size, stride):
            out_size = ((in_size - 1) * stride) + (k_size - 1) + 1 # formula: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html

            return out_size

        def check_sizes(in_size, k_size, stride):
            out_size = get_out_size(in_size, k_size, stride)
            while out_size > feature_size:
                k_size, stride = adapt_filter_size(in_size, k_size, stride)
                out_size = get_out_size(in_size, k_size, stride)

            return out_size, k_size, stride

        # default sizes of original architecture
        in_size_1, k_size_1, stride_1 = 1, 5, 1
        k_size_2, stride_2 = 5, 4
        k_size_3, stride_3 = 7, 4
        k_size_4, stride_4 = 7, 3
        k_size_5, stride_5 = 7, 2
        k_size_6, stride_6 = 3, 2

        # conv 1
        #print("\nConv1 before in={} k={} s={}:".format(in_size_1, k_size_1, stride_1))
        in_size_2, k_size_1, stride_1 = check_sizes(in_size_1, k_size_1, stride_1)
        #print("Conv1 after in={} k={} s={}:".format(in_size_1, k_size_1, stride_1))
        # conv 2
        #print("Conv2 before in={} k={} s={}:".format(in_size_2, k_size_2, stride_2))
        in_size_3, k_size_2, stride_2 = check_sizes(in_size_2, k_size_2, stride_2)
        #print("Conv2 after in={} k={} s={}:".format(in_size_2, k_size_2, stride_2))
        # conv 3
        #print("Conv3 before in={} k={} s={}:".format(in_size_3, k_size_3, stride_3))
        in_size_4, k_size_3, stride_3 = check_sizes(in_size_3, k_size_3, stride_3)
        #print("Conv3 after in={} k={} s={}:".format(in_size_3, k_size_3, stride_3))
        # conv  4
        #print("Conv4 before in={} k={} s={}:".format(in_size_4, k_size_4, stride_4))
        in_size_5, k_size_4, stride_4 = check_sizes(in_size_4, k_size_4, stride_4)
        #print("Conv4 after in={} k={} s={}:".format(in_size_4, k_size_4, stride_4))
        # conv 5
        #print("Conv5 before in={} k={} s={}:".format(in_size_5, k_size_5, stride_5))
        in_size_6, k_size_5, stride_5 = check_sizes(in_size_5, k_size_5, stride_5)
        #print("Conv5 after in={} k={} s={}:".format(in_size_5, k_size_5, stride_5))
        # conv 6
        #print("Conv6 before in={} k={} s={}:".format(in_size_6, k_size_6, stride_6))
        out_size_dec, k_size_6, stride_6 = check_sizes(in_size_6, k_size_6, stride_6)
        if out_size_dec < feature_size:
            stride_6 = 1
            k_size_6 = feature_size - in_size_6 + 1
            out_size_dec = ((in_size_6 - 1) * stride_6) + (k_size_6 - 1) + 1 # recovers any feature_size from any input size (as long as stride=1)
        #print("Conv6 after in={} k={} s={}:".format(in_size_6, k_size_6, stride_6))

        self.decoder = nn.Sequential(

            #Autoencoder2(),
            nn.ConvTranspose1d(in_channels=32 * n_channels_base, out_channels=16 * n_channels_base, kernel_size=k_size_1, stride=stride_1, padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
                         #     Autoencoder2(),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=k_size_2, stride=stride_2, padding=0,
                               dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
                     #      Autoencoder2(),
            nn.BatchNorm1d(8 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=8 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=k_size_3, stride=stride_3,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
                         #   Autoencoder2(),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=4 * n_channels_base, out_channels=2 * n_channels_base, kernel_size=k_size_4, stride=stride_4,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
                      #       Autoencoder2(),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=2 * n_channels_base, out_channels=n_channels_base, kernel_size=k_size_5, stride=stride_5,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
                       #       Autoencoder2(),
            #nn.BatchNorm1d(n_channels_base), # batch normalization transforms data in range close to 0 => hard to reconstruct the original data 
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=n_channels_base, out_channels=1, kernel_size=k_size_6, stride=stride_6,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
                          #    Autoencoder2(),
            #nn.Sigmoid(), the Sigmoid function creates values between 0-1
        )

        # self.decoder = nn.Sequential(nn.Linear(128, dataset_train_object.feature_size)
        #                              , nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x.view(-1, 1, x.shape[1]))
        x = self.decoder(x)
        x = torch.squeeze(x)
        # rounds discrete features
        for z in range(len(disc_feat)):
            if disc_feat.iloc[z] == 1:
                x[:,z] = torch.round(x[:,z])
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.genDim = 128
        self.linear1 = nn.Linear(opt.latent_dim, self.genDim)
        self.bn1 = nn.BatchNorm1d(self.genDim, eps=0.001, momentum=0.01)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(opt.latent_dim, self.genDim)
        self.bn2 = nn.BatchNorm1d(self.genDim, eps=0.001, momentum=0.01)
        self.activation2 = nn.Tanh()

    def forward(self, x):
        # Layer 1
        residual = x
        temp = self.activation1(self.bn1(self.linear1(x)))
        out1 = temp + residual

        # Layer 2
        residual = out1
        temp = self.activation2(self.bn2(self.linear2(out1)))
        out2 = temp + residual
        return out2

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Discriminator's parameters
        self.disDim = 256

        # The minibatch averaging setup
        ma_coef = 1
        if opt.minibatch_averaging:
            ma_coef = ma_coef * 2

        self.model = nn.Sequential(
            nn.Linear(ma_coef * dataset_train_object.feature_size, self.disDim),
            nn.ReLU(True),
            nn.Linear(self.disDim, int(self.disDim)),
            nn.ReLU(True),
            nn.Linear(self.disDim, int(self.disDim)),
            nn.ReLU(True),
            nn.Linear(int(self.disDim), 1)
        )

    def forward(self, x):

        if opt.minibatch_averaging:
            ### minibatch averaging ###
            x_mean = torch.mean(x, 0).repeat(x.shape[0], 1)  # Average over the batch
            x = torch.cat((x, x_mean), 1)  # Concatenation

        # Feeding the model
        output = self.model(x)
        return output


###############
### Lossess ###
###############

def autoencoder_loss(x_output, y_target):
    """
    autoencoder_loss
    This implementation is equivalent to the following:
    torch.nn.BCELoss(reduction='sum') / batch_size
    As our matrix is too sparse, first we will take a sum over the features and then do the mean over the batch.
    WARNING: This is NOT equivalent to torch.nn.BCELoss(reduction='mean') as the later on, mean over both features and batches.
    """
    MSE = nn.MSELoss()
    loss = MSE(x_output, y_target)
    return loss


#################
### Functions ###
#################

def discriminator_accuracy(predicted, y_true):
    """
    The discriminator accuracy on samples
    :param predicted: The predicted labels
    :param y_true: The gorund truth labels
    :return: Accuracy
    """
    total = y_true.size(0)
    correct = (torch.abs(predicted - y_true) <= 0.5).sum().item()
    accuracy = 100.0 * correct / total
    return accuracy


## was only needed for binary attributes
# def sample_transform(sample):
#     """
#     Transform samples to their nearest integer
#     :param sample: Rounded vector.
#     :return:
#     """
#     sample[sample >= 0.5] = 1
#     sample[sample < 0.5] = 0
#     return sample


def weights_init(m):
    """
    Custom weight initialization.
    :param m: Input argument to extract layer type
    :return: Initialized architecture
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


#############
### Model ###
#############

# Initialize generator and discriminator
generatorModel = Generator()
discriminatorModel = Discriminator()
autoencoderModel = Autoencoder()
autoencoderDecoder = autoencoderModel.decoder

# Define cuda Tensors
Tensor = torch.FloatTensor
one = torch.FloatTensor([1])
mone = one * -1


if torch.cuda.device_count() > 1 and opt.multiplegpu:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  generatorModel = nn.DataParallel(generatorModel, list(range(opt.num_gpu)))
  discriminatorModel = nn.DataParallel(discriminatorModel, list(range(opt.num_gpu)))
  autoencoderModel = nn.DataParallel(autoencoderModel, list(range(opt.num_gpu)))
  autoencoderDecoder = nn.DataParallel(autoencoderDecoder, list(range(opt.num_gpu)))

if opt.cuda:
    print("cuda in use")
    """
    model.cuda() will change the model inplace while input.cuda() 
    will not change input inplace and you need to do input = input.cuda()
    ref: https://discuss.pytorch.org/t/when-the-parameters-are-set-on-cuda-the-backpropagation-doesnt-work/35318
    """
    generatorModel.cuda()
    discriminatorModel.cuda()
    autoencoderModel.cuda()
    autoencoderDecoder.cuda()
    one, mone = one.cuda(), mone.cuda()
    Tensor = torch.cuda.FloatTensor

# Weight initialization
generatorModel.apply(weights_init)
discriminatorModel.apply(weights_init)
autoencoderModel.apply(weights_init)

# Optimizers
g_params = [{'params': generatorModel.parameters()},
            {'params': autoencoderDecoder.parameters(), 'lr': 1e-4}]
#g_params = list(generatorModel.parameters()) + list(autoencoderModel.decoder.parameters())
optimizer_G = torch.optim.Adam(g_params, lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
optimizer_D = torch.optim.Adam(discriminatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)
optimizer_A = torch.optim.Adam(autoencoderModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)

################
### TRAINING ###
################
if opt.training:

    if opt.resume:
        #####################################
        #### Load model and optimizer #######
        #####################################

        # Loading the checkpoint
        checkpoint = torch.load(os.path.join(opt.PATH, "model_epoch_1000.pth"))

        # Load models
        generatorModel.load_state_dict(checkpoint['Generator_state_dict'])
        discriminatorModel.load_state_dict(checkpoint['Discriminator_state_dict'])
        autoencoderModel.load_state_dict(checkpoint['Autoencoder_state_dict'])
        autoencoderDecoder.load_state_dict(checkpoint['Autoencoder_Decoder_state_dict'])

        # Load optimizers
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        optimizer_A.load_state_dict(checkpoint['optimizer_A_state_dict'])

        # Load losses
        g_loss = checkpoint['g_loss']
        d_loss = checkpoint['d_loss']
        a_loss = checkpoint['a_loss']

        # Load epoch number
        epoch = checkpoint['epoch']

        generatorModel.eval()
        discriminatorModel.eval()
        autoencoderModel.eval()
        autoencoderDecoder.eval()

    for epoch_pre in range(opt.n_epochs_pretrain):
        for i, samples in enumerate(dataloader_train):

            # Configure input
            real_samples = Variable(samples.type(Tensor))

            # Generate a batch of images
            recons_samples = autoencoderModel(real_samples)

            # Loss measures generator's ability to fool the discriminator
            a_loss = autoencoder_loss(recons_samples, real_samples)

            # Reset gradients
            optimizer_A.zero_grad()

            a_loss.backward()
            optimizer_A.step()

            batches_done = epoch_pre * len(dataloader_train) + i
            if batches_done % opt.sample_interval == 0:
                if opt.epoch_time_show:
                    print(
                        "[Epoch %d/%d of pretraining] [Batch %d/%d] [A loss: %.3f]"
                        % (epoch_pre + 1, opt.n_epochs_pretrain, i, len(dataloader_train), a_loss.item())
                        , flush=True)

    gen_iterations = 0
    for epoch in range(opt.n_epochs):
        epoch_start = time.time()
        for i, samples in enumerate(dataloader_train):

            # Adversarial ground truths
            valid = Variable(Tensor(samples.shape[0]).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(samples.shape[0]).fill_(0.0), requires_grad=False)

            # Configure input
            real_samples = Variable(samples.type(Tensor))

            # Sample noise as generator input
            z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            for p in discriminatorModel.parameters():  # reset requires_grad
                p.requires_grad = True

            # train the discriminator n_iter_D times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                n_iter_D = 100
            else:
                n_iter_D = opt.n_iter_D
            j = 0
            while j < n_iter_D:
                j += 1

                # clamp parameters to a cube
                for p in discriminatorModel.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                # reset gradients of discriminator
                optimizer_D.zero_grad()

                errD_real = torch.mean(discriminatorModel(real_samples), dim=0)
                errD_real = abs(1-errD_real) # goal is to identify as real (1)
                errD_real.backward() # Calculates the gradients of the NN parameters (discriminator model >> parameters set to require_grad=True)

                # Measure discriminator's ability to classify real from generated samples
                # The detach() method constructs a new view on a tensor which is declared
                # not to need gradients, i.e., it is to be excluded from further tracking of
                # operations, and therefore the subgraph involving this view is not recorded.
                # Refer to http://www.bnikolic.co.uk/blog/pytorch-detach.html.

                # Sample noise as generator input
                z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

                # Generate a batch of images
                fake_samples = generatorModel(z)

                # uncomment if there is no autoencoder
                fake_samples = torch.squeeze(autoencoderDecoder(fake_samples.unsqueeze(dim=2)))
                # rounds discrete features
                for z in range(len(disc_feat)):
                    if disc_feat.iloc[z] == 1:
                        fake_samples[:,z] = torch.round(fake_samples[:,z])

                errD_fake = torch.mean(discriminatorModel(fake_samples.detach()),dim=0)
                errD_fake = abs(0-errD_fake) # goal is to identify as fake (0)
                errD_fake.backward()
                errD = abs(errD_real + errD_fake)

                # Optimizer step
                optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------

            # We’re supposed to clear the gradients each iteration before calling loss.backward() and optimizer.step().
            #
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch
            # accumulates the gradients on subsequent backward passes. This is convenient while training RNNs. So,
            # the default action is to accumulate (i.e. sum) the gradients on every loss.backward() call.
            #
            # Because of this, when you start your training loop, ideally you should zero out the gradients so
            # that you do the parameter update correctly. Else the gradient would point in some other direction
            # than the intended direction towards the minimum (or maximum, in case of maximization objectives).

            # Since the backward() function accumulates gradients, and you don’t want to mix up gradients between
            # minibatches, you have to zero them out at the start of a new minibatch. This is exactly like how a general
            # (additive) accumulator variable is initialized to 0 in code.

            for p in discriminatorModel.parameters():  # reset requires_grad
                p.requires_grad = False
            
            for p in generatorModel.parameters():  # reset requires_grad
                p.requires_grad = True
            
            for p in autoencoderDecoder.parameters():  # reset requires_grad
                p.requires_grad = True
            
            # Zero grads
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

            # Generate a batch of images
            fake_samples = generatorModel(z)

            # uncomment if there is no autoencoder
            fake_samples = torch.squeeze(autoencoderDecoder(fake_samples.unsqueeze(dim=2)))
            # rounds discrete features
            for z in range(len(disc_feat)):
                if disc_feat.iloc[z] == 1:
                    fake_samples[:,z] = torch.round(fake_samples[:,z])

            # Loss measures generator's ability to fool the discriminator
            errG = torch.mean(discriminatorModel(fake_samples), dim=0)
            errG = abs(1-errG) # goal is to fool the NN to identify as real (1)
            errG.backward()

            # read more at https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/4
            optimizer_G.step()
            gen_iterations += 1

        with torch.no_grad():

            # Variables
            real_samples_test = next(iter(dataloader_test))
            real_samples_test = Variable(real_samples_test.type(Tensor))
            z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

            # Generator
            fake_samples_test_temp = generatorModel(z)
            fake_samples_test = torch.squeeze(autoencoderDecoder(fake_samples_test_temp.unsqueeze(dim=2)))
            # rounds discrete features
            for z in range(len(disc_feat)):
                if disc_feat.iloc[z] == 1:
                    fake_samples_test[:,z] = torch.round(fake_samples_test[:,z])

            # Discriminator
            # F.sigmoid() is needed as the discriminator outputs are logits without any sigmoid.
            out_real_test = discriminatorModel(real_samples_test).view(-1)
            accuracy_real_test = discriminator_accuracy(F.sigmoid(out_real_test), valid)

            out_fake_test = discriminatorModel(fake_samples_test.detach()).view(-1)
            accuracy_fake_test = discriminator_accuracy(F.sigmoid(out_fake_test), fake)

            # Test autoencoder
            reconst_samples_test = autoencoderModel(real_samples_test)
            a_loss_test = autoencoder_loss(reconst_samples_test, real_samples_test)

        
        if opt.epoch_time_show:
            print('TRAIN: [Epoch %d/%d] [Batch %d/%d] Loss_D: %.3f Loss_G: %.3f Loss_D_real: %.3f Loss_D_fake %.3f'
                % (epoch + 1, opt.n_epochs, i, len(dataloader_train),
                    errD.item(), errG.item(), errD_real.item(), errD_fake.item()), flush=True)

            print(
                "TEST: [Epoch %d/%d] [Batch %d/%d] [A loss: %.2f] [real accuracy: %.2f] [fake accuracy: %.2f]"
                % (epoch + 1, opt.n_epochs, i, len(dataloader_train),
                a_loss_test.item(), accuracy_real_test,
                accuracy_fake_test)
                , flush=True)

        # End of epoch
        epoch_end = time.time()
        if opt.epoch_time_show:
            print("It has been {0} seconds for this epoch".format(epoch_end - epoch_start), flush=True)

        if (epoch + 1) % opt.epoch_save_model_freq == 0:
            if os.path.exists(os.path.join(opt.expPATH, "models_epoch_%d" % (epoch + 1))) == False:
                os.mkdir(os.path.join(opt.expPATH, "models_epoch_%d" % (epoch + 1)))
            # Refer to https://pytorch.org/tutorials/beginner/saving_loading_models.html
            torch.save({
                'epoch': epoch + 1,
                'Generator_state_dict': generatorModel.state_dict(),
                'Discriminator_state_dict': discriminatorModel.state_dict(),
                'Autoencoder_state_dict': autoencoderModel.state_dict(),
                'Autoencoder_Decoder_state_dict': autoencoderDecoder.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'optimizer_A_state_dict': optimizer_A.state_dict(),
            }, os.path.join(opt.expPATH, "models_epoch_{}/cl_{}.pth".format((epoch + 1), opt.cl)))


if opt.finetuning:

    # Loading the checkpoint
    checkpoint = torch.load(os.path.join(opt.PATH, "model_epoch_100.pth"))

    # Setup model
    generatorModel = Generator()
    discriminatorModel = Discriminator()

    if cuda:
        generatorModel.cuda()
        discriminatorModel.cuda()
        discriminator_loss.cuda()

    # Setup optimizers
    optimizer_G = torch.optim.Adam(generatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Load models
    generatorModel.load_state_dict(checkpoint['Generator_state_dict'])
    discriminatorModel.load_state_dict(checkpoint['Discriminator_state_dict'])

    # Load optimizers
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    # Load losses
    g_loss = checkpoint['g_loss']
    d_loss = checkpoint['d_loss']

    # Load epoch number
    epoch = checkpoint['epoch']

    generatorModel.eval()
    discriminatorModel.eval()

if opt.generate:

    # Check cuda
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device BUT it is not in use...")

    # Activate CUDA
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    #####################################
    #### Load model and optimizer #######
    #####################################

    # Loading the checkpoint
    checkpoint = torch.load(os.path.join(opt.expPATH, "models_epoch_{}/cl_{}.pth".format(opt.n_epochs, opt.cl)))

    # Load models
    generatorModel.load_state_dict(checkpoint['Generator_state_dict'])
    autoencoderModel.load_state_dict(checkpoint['Autoencoder_state_dict'])
    autoencoderDecoder.load_state_dict(checkpoint['Autoencoder_Decoder_state_dict'])

    # insert weights [required]
    generatorModel.eval()
    autoencoderModel.eval()
    autoencoderDecoder.eval()

    #######################################################
    #### Load real data and generate synthetic data #######
    #######################################################

    # Load real data
    real_samples = dataset_train_object.return_data()
    num_fake_samples = round(sample_size * opt.smpl_frac) # size of original data

    # Generate a batch of samples
    gen_samples = np.zeros_like(np.zeros((num_fake_samples, feature_size)), dtype=type(real_samples))
    n_batches = int(num_fake_samples / opt.batch_size)
    for i in range(n_batches):
        # Sample noise as generator input
        z = torch.randn(opt.batch_size, opt.latent_dim, device=device)
        gen_samples_tensor = generatorModel(z)
        gen_samples_decoded = torch.squeeze(autoencoderDecoder(gen_samples_tensor.unsqueeze(dim=2)))
        # rounds discrete features
        for z in range(len(disc_feat)):
            if disc_feat[z] == 1:
                gen_samples_decoded[:,z] = torch.round(gen_samples_decoded[:,z])
        gen_samples[i * opt.batch_size:(i + 1) * opt.batch_size, :] = gen_samples_decoded.cpu().data.numpy()
        # Check to see if there is any nan
        assert (gen_samples[i, :] != gen_samples[i, :]).any() == False
    gen_samples = np.delete(gen_samples, np.s_[(i + 1) * opt.batch_size:], 0)

    # Trasnform Object array to float
    gen_samples = gen_samples.astype(np.float32)

    # save synthetic data
    np.save(os.path.join(opt.expPATH, "synthetic.npy"), gen_samples, allow_pickle=False)

