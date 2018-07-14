import numpy as np
import pandas as pd
import tensorflow as tf # pylint: disable=import-error
from sklearn.model_selection import KFold


PATH = '../input/'

LR = 0.1

BATCH_SIZE = 50

N_SAMPLES = len(list(open(f'{PATH}train.csv'))) - 1 # skip header

STEPS = np.ceil(N_SAMPLES // BATCH_SIZE)

HIDDEN_UNITS = [10, 10]

N_CLASSES = 2

L1_REGULARIZATION_STRENGTH = 0.001

KFOLD_SPLIT_SIZE = 5


def get_feature_columns():
    numeric_column_age = tf.feature_column.numeric_column(key='Age')

    return [
        numeric_column_age
    ]


def train_input_fn(features, labels):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(BATCH_SIZE)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert BATCH_SIZE is not None, "BATCH_SIZE must not be None"
    dataset = dataset.batch(BATCH_SIZE)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def run_train(classifier, train_x, train_y):
    train_samples = len(train_x['Age'])
    print(
        f'All shapes - train_x: {train_samples}, train_y: {len(train_y)}'
    )
    classifier.train(
        input_fn=lambda: train_input_fn(train_x, train_y),
        steps=STEPS)


def cross_validate(classifier, train_x_all, train_y_all):
    print(
        f'All shapes - train_x_all: {len(train_x_all)}, train_y_all: {len(train_y_all)}'
    )
    results = []
    kf = KFold(n_splits=KFOLD_SPLIT_SIZE)
    for train_idx, val_idx in kf.split(train_x_all, train_y_all):
        train_x = {'Age': train_x_all[train_idx]}
        train_y = train_y_all[train_idx]
        val_x = {'Age': train_x_all[val_idx]}
        val_y = train_y_all[val_idx]

        run_train(classifier, train_x, train_y)

        eval_result = classifier.evaluate(
            input_fn=lambda: eval_input_fn(val_x, val_y))
        results.append(eval_result)

    return results


def main():
    train = pd.read_csv(f'{PATH}train.csv')
    print('train shape:', train.shape)
    train.head()

    test = pd.read_csv(f'{PATH}test.csv')
    print('test shape:', test.shape)
    test.tail()

    classifier = tf.estimator.DNNClassifier(
        feature_columns=get_feature_columns(),
        # Two hidden layers of 10 nodes each.
        hidden_units=HIDDEN_UNITS,
        # The model must choose between 3 classes.
        n_classes=N_CLASSES,
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=LR,
            l1_regularization_strength=L1_REGULARIZATION_STRENGTH
        ))

    train_x_all = train['Age'].fillna(0)
    train_y_all = train['Survived']
    results = cross_validate(classifier, train_x_all, train_y_all)
    for r in results:
        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**r))


if __name__ == '__main__':
    main()
