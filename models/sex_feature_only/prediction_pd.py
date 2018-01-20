# model prediction using Pandas and 'tf.estimator.inputs.numpy_input_fn'
import pandas as pd

test_df = pd.read_csv('input/test.csv')
pall_pdf = np.array([test_df['Sex']])

print(pall_pdf.T.shape)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"Sex": pall_pdf.T},
    num_epochs=1,
    shuffle=False)

predictions = model.predict(input_fn=predict_input_fn)
predicted_classes = [p["classes"][0].decode('utf8') for p in predictions]
sum((int(x) for x in predicted_classes))

submission = pd.DataFrame(data={'PassengerId': test_df['PassengerId'], 'Survived': predicted_classes})
submission.to_csv('input/submission.csv', index=False)
submission.tail()