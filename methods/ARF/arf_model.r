# python implementation available: https://pypi.org/project/arfpy/


# libraries
library(data.table)
library(ranger)
library(foreach)
library(truncnorm)
library(matrixStats)
library(arf)
#print(.libPaths())


suppressWarnings({

# Reads arguments from terminal
args <- commandArgs(trailingOnly = TRUE)
smpl_size <- args[1]

# load data
train_data <- read.csv('methods/ARF/train_data.csv', row.names = 1)

# Train the ARF
arf <- adversarial_rf(train_data, verbose = FALSE) # for discriminator accuracies change verbose to TRUE

# Estimate distribution parameters
psi <- forde(arf, train_data)

# synthesis
gen_data <- forge(psi, smpl_size)

# save fake data on disk
write.csv(gen_data, file = "methods/ARF/gen_data.csv", row.names = FALSE)

})