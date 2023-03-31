import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Imports for LR and RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Imports for NN
import tensorflow as tf
from tensorflow import keras # make sure you have keras installed
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score


def make_models(X_train):  
  '''
  We will use four different models, which are initialized in the `make_models` function below.

  LR_L2: is a logistic regression with an L2 loss 
  LR_L1: is a logistic regression with an L1 loss with "balanced" class weights
  RF: is a random forest with "balanced" class weights
  ANN: is a Neural Network with a binary_crossentropy loss and adam optimizer
  '''
  return {
      'LR_L2': LogisticRegression(random_state=0, class_weight='balanced'),
      'LR_L1': LogisticRegression(random_state=0, penalty='l1', solver='liblinear', class_weight='balanced'),
      'RF': RandomForestClassifier(random_state=0, class_weight='balanced'),
      'ANN': keras.Sequential([
      keras.layers.Flatten(input_shape=(X_train.shape[1],)),
      keras.layers.Dense(2048, activation=tf.nn.relu),
      keras.layers.Dense(512, activation=tf.nn.relu),
      keras.layers.Dense(64, activation=tf.nn.relu),
      keras.layers.Dense(1, activation=tf.nn.sigmoid),])
  }

def fit_and_score_model(all_models, X_train, X_test, y_train, y_test, X_val, y_val, threshold):
    """Fits the models that are initialized by models_dict on the X_train and y_train
    data, and evaluates the model on the out-of-sample data X_out_of_sample and y_out_of_sample"""
    
    # Make a dictionary of the models
    models_dict = make_models(X_train)

    # Loop through each model in model_dict
    for model_name in models_dict:
        model = models_dict[model_name]

        if model_name == 'ANN':
          model.compile(optimizer='adam',loss='binary_crossentropy', 
                         metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
          
          yhat_probs = model.predict(X_test, verbose=1)
          yhat_classes = (model.predict(X_test) > 0.5).astype("int32")
          yhat_probs = yhat_probs[:, 0]
          yhat_classes = yhat_classes[:, 0]
          
          model_precision = precision_score(y_test, yhat_classes)
          model_recall = recall_score(y_test, yhat_classes)
          model_score = (model_precision + model_recall) / 2

        else:
          model.fit(X_train, y_train) 
          y_temp = model.predict_proba(X_test)
          y_pred = []
          for x in y_temp:
            if x[1] > threshold:
              y_pred.append(1)
            else:
              y_pred.append(0)

          model_precision = precision_score(y_test, y_pred)
          model_recall = recall_score(y_test, y_pred)
          model_score = (model_precision + model_recall) / 2
          model_roc_auc =  roc_auc_score(y_test, y_pred)

        print(f'{model_name} achieved a precision of {model_precision:.3f} and recall of {model_recall:.3f}.{model_name} achieved a ROC_AUC of {model_roc_auc:.3f} and score of {model_score:.3f}.')
        
        all_models.loc[model_name] = np.array((model_precision, model_recall, model_score, model_roc_auc, model), dtype='object')

    return all_models

def create_train_test_val(df_injury):
    properties = list(df_injury.columns.values)
    properties.remove('currently_injured')
    X = df_injury[properties]
    y = df_injury['currently_injured']

    # Remove Outliers
    isf = IsolationForest(n_jobs=-1, random_state=1)
    isf.fit(X, y)
    preds = isf.predict(X)

    X['outlier'] = preds
    X = X.drop(X[X['outlier'] == -1].index)
    X = X.drop('outlier', axis=1)

    y = pd.DataFrame(y)
    y['outlier'] = preds
    y = y.drop(y[y['outlier'] == -1].index)
    y = y.drop('outlier', axis=1)
    y = y['currently_injured'].squeeze()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, random_state=1)

    return X_train, X_test, X_val, y_train, y_test, y_val

def main():
    
    # Data Import
    import os
    cwd = os.getcwd()

    X_train=pd.read_csv(os.path.join(cwd,'/final_datasets/X_train.csv'))
    X_test=pd.read_csv(os.path.join(cwd,'/final_datasets/X_test.csv'))
    X_val=pd.read_csv(os.path.join(cwd,'/final_datasets/X_val.csv'))
    y_train=pd.read_csv(os.path.join(cwd,'/final_datasets/y_train.csv'))
    y_test=pd.read_csv(os.path.join(cwd,'/final_datasets/y_test.csv'))
    y_val=pd.read_csv(os.path.join(cwd,'/final_datasets/y_val.csv'))

    # Create a data frame to keep track of all the models we train
    model_names = ('LR_L2', 'LR_L1', 'RF', 'ANN')

    # Initialize the `all_models` data frame
    all_models = pd.DataFrame(index=model_names, columns=['Precision', 'Recall', 'Score', 'ROC AUC', 'Model'])
    all_models[['Precision', 'Recall', 'Score', 'ROC AUC']] = all_models[['Precision', 'Recall', 'Score', 'ROC AUC']].astype(float)

    # Run all models
    all_models = fit_and_score_model(all_models, X_train, X_test, y_train, y_test, X_val, y_val, 0.5)

    # Get final scores in csv file
    all_models.to_csv(os.path.join(cwd,'model_performance.csv'))

if __name__ == '__main__':
   main()
