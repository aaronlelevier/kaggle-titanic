# model.predict with TF
def tf_predict_input_fn(data_file):

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=[['']])
        features = dict(zip(['Sex'], columns))
        return features

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    
    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(1) # times to repeat
    dataset = dataset.batch(1) # batch size - NOTE: probably ignored since "repeat=1"

    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features

predictions = model.predict(input_fn=lambda: tf_predict_input_fn(
        data_file='input/test-sex.csv'))
predicted_classes = [p["classes"][0].decode('utf8') for p in predictions]
submission = pd.DataFrame(data={'PassengerId': test_df['PassengerId'], 'Survived': predicted_classes})
submission.to_csv('input/submission-tf.csv', index=False)
submission.tail()