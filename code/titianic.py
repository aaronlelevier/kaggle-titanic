from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

PATH = '../input/'

train = pd.read_csv(f'{PATH}train.csv')
print('train shape:', train.shape)
train.head()

test = pd.read_csv(f'{PATH}test.csv')
print('test shape:', test.shape)
test.tail()

# Parameters
learning_rate = 0.01
training_epochs = 500 # 1000 - orig value
display_step = 50

batch_size = 50
train_steps = batch_size * 10

n_samples = train.shape[0]

train_split = .8
index_split = int(n_samples*train_split)
train_slice = slice(0, index_split)
test_slice = slice(index_split, n_samples)

train[train_slice].shape[0] + train[test_slice].shape[0] == train.shape[0] 

# Age as the only feature
print('train Age is NaN:', len(train[train['Age'].isnull()]))
print('test Age is NaN:', len(test[test['Age'].isnull()]))


# separated train and dev sets
train_set = train[train_slice]
dev_set = train[test_slice]

# Train / Dev set splits
# train
train_Y = train_set['Survived']
train_X = train_set['Age'].fillna(0)

train_x = {'Age': train_X}
train_y = train_Y

# test
dev_Y = dev_set['Survived']
dev_X = dev_set['Age'].fillna(0)

dev_x = {'Age': dev_X}
dev_y = dev_Y


# Feature Columns
numeric_column_age = tf.feature_column.numeric_column(key='Age')
my_feature_columns = [
    numeric_column_age
]


classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=2)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


classifier.train(
    input_fn=lambda: train_input_fn(train_x, train_y, batch_size),
    steps=train_steps)


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


eval_result = classifier.evaluate(
    input_fn=lambda: eval_input_fn(dev_x, dev_y, batch_size))


print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


# ## Next
# 
# add more features
# 
# try in a script format, instead of NB
