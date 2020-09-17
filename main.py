from utils import *


# Number of folds for KFold validation strategy
FOLDS = 5

# Number of epochs to train each model
EPOCHS = 130

# Batch size
BATCH_SIZE = 64

# Learning rate
LR = 0.001

# Verbosity
VERBOSE = 1

# Seed for deterministic results
SEED = 123

if __name__ == "__main__":
    train_inputs, public_test, private_test,\
    train_img, test_public_img, test_private_img = get_sets('train.json', 'test.json', 
                                                            'sample_submission.csv',
                                                            'bpps/')
        