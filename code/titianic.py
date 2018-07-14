import argparse
import time

import numpy as np
import pandas as pd
import tensorflow as tf # pylint: disable=import-error
from sklearn.model_selection import KFold

# File Args
parser = argparse.ArgumentParser()
parser.add_argument('--predict', dest='predict', action='store_true')
parser.add_argument('--no-predict', dest='predict', action='store_false')
parser.set_defaults(predict=True)
args = parser.parse_args()


# Hyper parameters
LR = 0.1

EPOCHS = 1

BATCH_SIZE = 50

HIDDEN_UNITS = [10]

L1_REGULARIZATION_STRENGTH = 0.001

KFOLD_SPLIT_SIZE = 5

# Constants
PATH = '../input/'

N_SAMPLES = len(list(open(f'{PATH}train.csv'))) - 1 # skip header

STEPS = np.ceil(N_SAMPLES // BATCH_SIZE)

N_CLASSES = 2


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


def cross_validate(classifier, train_x_all, train_y_all):
    """
    Main function call for training and cross validating

    Args:
        classifier (tf.estimator.Estimator)
        train_x_all (pd.DataFrame)
        train_y_all (np.ndarray)
    """
    results = []
    kf = KFold(n_splits=KFOLD_SPLIT_SIZE)

    for epoch in range(EPOCHS):
        print(f'epoch: {epoch}')

        for train_idx, val_idx in kf.split(train_x_all, train_y_all):
            train_x = train_x_all.iloc[train_idx]
            train_y = train_y_all[train_idx]

            val_x = train_x_all.iloc[val_idx]
            val_y = train_y_all[val_idx]

            classifier.train(
                input_fn=lambda: train_input_fn(train_x, train_y),
                steps=STEPS)

            eval_result = classifier.evaluate(
                input_fn=lambda: eval_input_fn(val_x, val_y))
            results.append(eval_result)

    return results


def get_unique_family_sizes(train, test):
    train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
    test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
    return sorted(list(set(list(test['FamilySize'].values) + list(train['FamilySize'].values))))


def get_feature_columns(train, test):
    """
    Returns a list of tf feature_columns's

    Args:
        train (pd.DataFrame)
        test (pd.DataFrame)
    """
    numeric_column_age = tf.feature_column.numeric_column(key='Age')

    categorical_column_has_cabin = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            key='HasCabin', vocabulary_list=(0, 1)))

    categorical_column_family_size = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            key='FamilySize', vocabulary_list=get_unique_family_sizes(train, test)),
        dimension=2)

    categorical_column_has_cabin = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            key='IsAlone', vocabulary_list=(0, 1)))

    return [
        numeric_column_age,
        categorical_column_has_cabin,
        categorical_column_family_size
    ]


def get_features(df):
    "Returns a feature engineeered DataFrame"
    # FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # IsAlone
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    return {
        'Age': df['Age'].fillna(0),
        'HasCabin': df["Cabin"].apply(lambda x: 0 if type(x) == float else 1),
        'FamilySize': df['FamilySize'],
        'IsAlone': df['IsAlone']
    }


def make_predictions(classifier, test):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=get_features(test),
        num_epochs=1,
        shuffle=False)
    predictions = classifier.predict(input_fn=predict_input_fn)
    predicted_classes = [p["classes"][0].decode('utf8') for p in predictions]
    submission = pd.DataFrame(data={'PassengerId': test['PassengerId'], 'Survived': predicted_classes})
    submission.to_csv(f'{PATH}submission.csv', index=False)


def main():
    start_time = time.time()

    train = pd.read_csv(f'{PATH}train.csv')
    print('train shape:', train.shape)
    train.head()

    test = pd.read_csv(f'{PATH}test.csv')
    print('test shape:', test.shape)
    test.tail()

    classifier = tf.estimator.DNNClassifier(
        feature_columns=get_feature_columns(train, test),
        # Two hidden layers of 10 nodes each.
        hidden_units=HIDDEN_UNITS,
        # The model must choose between 3 classes.
        n_classes=N_CLASSES,
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=LR,
            l1_regularization_strength=L1_REGULARIZATION_STRENGTH
        ))

    train_x_all = pd.DataFrame(get_features(train))

    train_y_all = train['Survived']
    results = cross_validate(classifier, train_x_all, train_y_all)
    for r in results:
        print('\nTest set accuracy: {accuracy:0.4f}'.format(**r))

    if args.predict:
        print('make_predictions')
        make_predictions(classifier, test)
    else:
        print('NOT make_predictions')

    print("time: {seconds} seconds".format(seconds=(time.time() - start_time)))


if __name__ == '__main__':
    main()
