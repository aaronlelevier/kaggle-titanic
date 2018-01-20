csv_columns = ['Survived', 'Sex']
record_defaults = [[0], ['']]
num_examples = {
    'train': 700,
    'test': 891-700
}

run_config = tf.estimator.RunConfig().replace(
    session_config=tf.ConfigProto(device_count={'GPU': 0}))

def build_model_columns():
    sex = tf.feature_column.categorical_column_with_vocabulary_list(
        'Sex', ['male', 'female'])
    return [sex]

def build_estimator():
    feature_columns = build_model_columns()

    return tf.estimator.LinearClassifier(
        model_dir='/tmp/titanic/',
        feature_columns=feature_columns,
        config=run_config
    )

def input_fn(data_file, num_epochs, shuffle, batch_size):

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=record_defaults)
        features = dict(zip(csv_columns, columns))
        labels = features.pop('Survived')
        return features, tf.equal(labels, 1)

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=num_examples['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

print("loading, " + str(file_length) + " line(s)\n")

# Main "model" training
train_epochs = 40
epochs_per_eval = 2
batch_size = 40
train_data = 'input/train-sex-training-set.csv'
test_data = 'input/train-sex-dev-set.csv'

model = build_estimator()

for n in range(train_epochs // epochs_per_eval):
    model.train(input_fn=lambda: input_fn(
        train_data, epochs_per_eval, True, batch_size))

    results = model.evaluate(input_fn=lambda: input_fn(
        test_data, 1, False, batch_size))

    # display results

    print('Results at epoch', (n + 1) * epochs_per_eval)
    print('-' * 60)

    for key in sorted(results):
        print('%s: %s' % (key, results[key]))
