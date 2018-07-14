# [Kaggle Titanic competition](https://www.kaggle.com/c/titanic) repo

Got in the 50% percentile of 10k participants with this repo code using vanialla TensorFlow with some feature engineering running under 5 min locally on a CPU.

Uses [TensorFlow](https://www.tensorflow.org)

Main file is [code/titanic.py](https://github.com/aaronlelevier/kaggle-titanic/blob/master/code/titianic.py)

Feature engineering from [Sina](https://www.kaggle.com/sinakhorami)'s Kaggle Notebook [Titanic best working Classifier](https://www.kaggle.com/sinakhorami/titanic-best-working-classifier)

Demonstrates:

- using TensorFlow's `feature_columns`
- `DNNClassifier` estimator
- putting main loop in a `main` function
- propery separating "hyperparams" and "static" constants as globals
- Python's `argparse`
- `Pandas`
- `sklearn.KFold`
