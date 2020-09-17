import os
import pandas as pd
import numpy as np
import seaborn as sns
import math
import random
import json
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics

# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    
# Preprocesing function to transform features to 3d format
def preprocess_inputs(df, token2int, cols = ['sequence', 'structure', 'predicted_loop_type']):
    return np.transpose(
        np.array(
            df[cols]\
            .applymap(lambda seq: [token2int[x] for x in seq])\
            .values\
            .tolist()
        ),
        (0, 2, 1)
    )
        
def get_sets(train_path, test_path, submission_path, images_path):
    # Read training, test and sample submission data
    train = pd.read_json(train_path, lines = True)
    test = pd.read_json(test_path, lines = True)
    sample_sub = pd.read_csv(submission_path)
    
    print('#GETTING FEATURES DATA#\n')
    
    # Target column list
    target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    # Dictionary comprehension to map the token with an specific id
    token2int = {x: i for i, x in enumerate('().ACGUBEHIMSX')}
    
    # Get useful samples
    train = train.loc[train['SN_filter'] == 1]
    
    # Transform training feature sequences to a 3d matrix of (x, 107, 3)
    train_inputs = preprocess_inputs(train, token2int)
    
    # Transform training targets sequences to a 3d matrix of (x, 68, 5)
    train_labels = np.array(train[target_cols].values.tolist()).transpose(0, 2, 1)
    
    # Get different test sets
    public_test_df = test[test['seq_length'] == 107]
    private_test_df = test[test['seq_length'] == 130]
    
    # Preprocess the test sets to the same format as our training data
    public_test = preprocess_inputs(public_test_df, token2int)
    private_test = preprocess_inputs(private_test_df, token2int)
    
    print('Train shape ', train_inputs.shape)
    print('Public test shape ', public_test.shape)
    print('Private test shape ', private_test.shape)
    
    print('#GETTING IMAGES DATA#\n')
        
    # Get train images 
    train_data_ids = train['id'].values
    test_public_ids = public_test_df['id'].values
    test_private_ids = private_test_df['id'].values

    train_img = []
    for ID in train_data_ids:
        img_path = os.path.join(images_path, ID+'.npy')
        img = np.load(img_path)
        train_img.append(img)
    
    test_public_img = []
    for ID in test_public_ids:
        img_path = os.path.join(images_path, ID+'.npy')
        img = np.load(img_path)
        test_public_img.append(img)
    
    test_private_img = []
    for ID in test_private_ids:
        img_path = os.path.join(images_path, ID+'.npy')
        img = np.load(img_path)
        test_private_img.append(img)    
    
    # Get tensors
    train_img = np.array(train_img)
    test_public_img = np.array(test_public_img)
    test_private_img = np.array(test_private_img)
    
    print('Train images shape ', train_img.shape)
    print('Public images test shape ', test_public_img.shape)
    print('Private images test shape ', test_private_img.shape)
    
    return train_inputs, public_test, private_test, train_img, test_public_img, test_private_img 
    
    