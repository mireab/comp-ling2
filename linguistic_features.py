# Importo le librerie necessarie
import os
import re
import shutil
import zipfile
import requests

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk import FreqDist
from nltk.util import ngrams
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

out_dir = 'Processing UD/input_profilingUD'

def preprocessing(input_path):
    """ Dato il file in formato tsv fornito per il task Haspeede20 estrae i dati (id, testo, labels) e crea per ogni testo
    un file .txt che posiziona nella cartella indicata. Questo è fondamentale per passare i files a Profiling-UD"""
    os.makedirs(out_dir, exist_ok=True)
    with open (in_file, 'r') as infile:
        lines = infile.readlines()
        for line in lines[1:]:
            row = line.strip().split('\t')
            ident = row[0]
            testo = row[1]
            hs = row[2]
            stereotype = row[3]
            out_file_title = f"{ident},{hs},{stereotype}"
            if 'train' in in_file:
              out_file = open( f"{out_dir}/train-{out_file_title}.txt", "a+")
            if 'test' in in_file:
              out_file = open( f"{out_dir}/test-{out_file_title}.txt", "a+")
            out_file.write(f"{testo}")

in_file = 'haspeede2_dev/haspeede2_train_taskAB.tsv'
preprocessing(in_file)

in_file = 'haspeede2_dev/haspeede2_test_taskAB-tweets.tsv'
preprocessing(in_file)

# Comprimo la cartella da passare a profiling_UD
directory_to_zip = 'Processing UD/input_profilingUD'
shutil.make_archive('Processing UD/input_profilingUD', 'zip', directory_to_zip)

profiling_output_path = 'Processing UD/9437.csv' # Path relativo della cartella di output di profiling_UD

def build_dataset(path):
    """ Funzione che, data la cartella di output di Profiling_UD, costruisce i dataset con le features estratte"""
    dataset_train = []
    dataset_test = []
    for line in open(path, 'r'):
        splitted_line = line.strip().split('\t')
        if splitted_line[0] == 'Filename':
            features_names = splitted_line[:]
        else:
            # Process the first column by removing the prefix
            first_column = splitted_line[0]
            # Extract the three 4numbers separated by commas
            classes = [int(num) for num in re.findall(r'\d+', first_column)]
            # Extract hs and stereotypes from the numbers list
            hs = classes[1]
            stereotypes = classes[2]

            # Reduce the first column to <first number>.conllu
            first_column_reduced = f"{classes[0]}"

            # Insert the reduced first column and the other features in the row
            new_row = [first_column_reduced, *splitted_line[1:], hs, stereotypes]
            if 'train' in first_column:
              dataset_train.append(new_row)
            if 'test' in first_column:
              dataset_test.append(new_row)

    return dataset_train, dataset_test, features_names

dataset_train, dataset_test, features_names = build_dataset(profiling_output_path)


features_names = features_names + ['hs','stereotypes']
train = pd.DataFrame(dataset_train, columns=features_names)

hs_labels = train['hs']
stereotypes_labels = train['stereotypes']
train_data = train.iloc[:, 1:-2] # Rimuovo Filename e labels

def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
  """ Funzione che restituisce i plot delle accuracy ottenute con le varie combinazioni di parametri """
  scores_mean = cv_results['mean_test_score']
  scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

  scores_sd = cv_results['std_test_score']
  scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

  # Plot Grid search scores
  _, ax = plt.subplots(1,1)

  # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
  for idx, val in enumerate(grid_param_2):
      ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

  ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
  ax.set_xlabel(name_param_1, fontsize=16)
  ax.set_ylabel('CV Average Score', fontsize=16)
  ax.legend(loc="best", fontsize=15)
  ax.grid('on')

def run_experiment(train, labels):
  """ Funzione che normalizza il dataset, ricerca gli iperparametri migliori per il LinearSVC tramite GridSearch, effettua una 5fold cross validation, stampa i 
  risultati e la confusion matrix e restituisce lo scaler e il modello ottenuto, da usare per il test """
  scaler = MinMaxScaler()
  X_train = scaler.fit_transform(train)
  C = [0.1, 1.0, 3.0, 10.0]
  dual = [True, False]
  svc = LinearSVC(max_iter=15000)

  grid_search = GridSearchCV(svc, 
                             dict(C = C, dual = dual), 
                 cv=5, n_jobs=-1, verbose=1)
  grid_search.fit(X_train, labels)

  plot_grid_search(grid_search.cv_results_, C, dual, 'C', 'Dual')
  best_params = grid_search.best_params_
  print("Best Hyperparameters:", best_params)

  svc = LinearSVC(C=best_params['C'], dual=best_params['dual'], max_iter=15000)
  svc.fit(X_train, labels)

  train_predictions = svc.predict(X_train)
  print(classification_report(labels, train_predictions, zero_division=0))

  num_folds = 5
  kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
  accuracy_scores = cross_val_score(svc, X_train, labels, cv=kf)

  for fold, accuracy in enumerate(accuracy_scores, start=1):
      print(f"Fold {fold}: Accuracy = {accuracy:.2f}")

  mean_accuracy = accuracy_scores.mean()
  std_accuracy = accuracy_scores.std()
  print(f"\nMean Accuracy: {mean_accuracy:.2f}")
  print(f"Standard Deviation: {std_accuracy:.2f}")

  ConfusionMatrixDisplay.from_predictions(labels, train_predictions, xticks_rotation='vertical', cmap='Blues')
  return scaler, svc


test = pd.DataFrame(dataset_test, columns=features_names)
test_data = test.iloc[:, 1:-2] 
y_test = test['hs']


# Addestro un DummyClassifier da utilizzare come baseline per il modello
print("Prestazione di uno zeroR classifier")
classifier = DummyClassifier(strategy="most_frequent")
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_data)
classifier.fit(X_train, hs_labels)
X_test = scaler. transform(test_data)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred, zero_division=0))


# Test del modello
# Creazione dataset da output Profiling UD
print("TEST DEL MODELLO PER HS")
# Carico lo scaler ed il modello addestrato
scaler, svc_hs = run_experiment(train_data, hs_labels)
test = pd.DataFrame(dataset_test, columns=features_names)
test_data = test.iloc[:, 1:-2] #Rimuovo 'Filename' e le labels

X_test = scaler.transform(test_data)
y_test = test['hs']
test_predictions = svc_hs.predict(X_test)

print(classification_report(y_test, test_predictions, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test, test_predictions, xticks_rotation='vertical', cmap='Blues');


"""#### Features più importanti"""
svc_hs.coef_
features_names = list(train_data.columns)

def f_importances(coef, names, top_n=15):
    
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    imp = imp[-top_n:]  # Select the top 15 most important features
    names = names[-top_n:]  # Corresponding feature names
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

f_importances(svc_hs.coef_[0], features_names, 15)




