import os
import shutil
import zipfile
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import regex as re
from nltk import FreqDist
from nltk.util import ngrams
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

zip_file_path = "9437.zip"
destination_dir = "annotated_docs"

# Questa parte non è necessaria se i files sono già stati estratti dalla
# cartella.

# with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
#     zip_ref.extractall(destination_dir)


source_dir = "annotated_docs"
# destination_dir = "annotated_docs"
# os.makedirs(destination_dir, exist_ok=True)
# for root, _, files in os.walk(source_dir):
#     for file in files:
#         source_file_path = os.path.join(root, file)
#         shutil.move(source_file_path, os.path.join(destination_dir, file))

file_list = os.listdir("annotated_docs")

# Creo le liste che saranno riempite con i files di addestramento  e di test
train_files = []
test_files = []

for file_name in file_list:
    file_path = os.path.join(source_dir, file_name)
    if "train" in file_name:
        train_files.append(file_path)
    elif "test" in file_name:
        test_files.append(file_path)


def get_sentences_from_file(src_path):
    """ Funzione che estrae dal file il documento, nella forma di una lista di liste, dove le sottoliste 
    contengono le parole del singolo tweet. Le parole sono rappresentate da dizionari contenenti 'word', 
    'lemma', 'pos'"""
    doc_sentences = []
    sentence = []
    for line in open(src_path, "r"):
        if line[0].isdigit():
            splitted_line = line.strip().split("\t")
            if (
                "-" not in splitted_line[0]
            ):  # se l'id della parola non contiene un trattino
                token = {
                    "word": splitted_line[1],
                    "lemma": splitted_line[2],
                    "pos": splitted_line[3],
                }
                sentence.append(token)
        if line == "\n":  # se la riga è vuota significa che la frase è finita
            doc_sentences.append(sentence)
            sentence = []
    return doc_sentences


train_dataset = [get_sentences_from_file(doc_path) for doc_path in train_files]



def extract_ngrams(document, element, n):
    """ Funzione che estrae da un documento ngrammi (dove n è argomento) di parole, pos o lemmi 
    (specificabili come elemento) e restituisce un dizionario di ngrammi-frequenze e la lunghezza 
    del documento """
    document_words = []
    doc_ngrams = []
    freq = []
    for sentence in document:
        sent_words = []
        for word in sentence:
            sent_words.append(word[element])
        document_words.append(sent_words)
    for sentence_words in document_words:
        ngrams_result = list(ngrams(sentence_words, n))
        doc_ngrams += ngrams_result
        freq = dict(FreqDist(doc_ngrams))
    doc_len = sum(len(sentence) for sentence in document_words)
    return freq, doc_len # La lunghezza sarà comoda successivamente


def extract_char_ngrams(document, n):
    """ Funzione che, dato un documento, estrae ngrammi di caratteri (dove n è argomento) restituendo un dizionario 
    di frequenze di ngrammi di caratteri e la lunghezza del documento """
    document_words = []
    document_ngrams = []
    all_words = ""
    for sentence in document:
        sentence_char_ngrams = []
        sent_words = []
        for word in sentence:
            sent_words.append(word["word"])
        document_words.append(
            sent_words
        ) 
    for sentence_words in document_words:  
        words = " ".join(sentence_words)
        for i in range(0, len(words) - n + 1):
            ngram = words[i : i + n]
            sentence_char_ngrams.append(tuple(ngram))
        all_words += words
    document_ngrams += sentence_char_ngrams
    freq = dict(FreqDist(document_ngrams))
    length = len(all_words)
    return freq, length


def normalize(ngrams):
    """ Funzione che normalizza le frequenze degli ngrammi restituendo il dizionario di"""
    frequencies, length = ngrams
    return {ngram: freq / length for ngram, freq in frequencies.items()}


def extract_features(dataset):
    """ Funzione che estrae le features, modificata di esperimento in esperimento in base alle features che si preferiscono """
    dataset_ft = []
    for document in dataset:
        unigrams_word = normalize(extract_ngrams(document, "word", 1))
        bigramns_word = normalize(extract_ngrams(document, "word", 2))
        trigrams_word = normalize(extract_ngrams(document, "word", 3))
        unigrams_pos = normalize(extract_ngrams(document, "pos", 1))
        bigramns_pos = normalize(extract_ngrams(document, "pos", 2))
        trigrams_pos = normalize(extract_ngrams(document, "pos", 3))
        unigrams_lemma = normalize(extract_ngrams(document, "lemma", 1))
        bigramns_lemma = normalize(extract_ngrams(document, "lemma", 2))
        trigrams_lemma = normalize(extract_ngrams(document, "lemma", 3))
        dataset_ft.append(
            unigrams_word
            | bigramns_word
            | trigrams_word
            | unigrams_pos
            | bigramns_pos
            | trigrams_pos
            | unigrams_lemma
            | bigramns_lemma
            | trigrams_lemma
        )
    return dataset_ft



def filter_features(train_features, min_occurrences):
    """ Funzione che filtra le features con la possibilità di fissare un numero minimo di occorrenze affinché la 
     feature venga impiegata """
    features_counter = Counter(
        feature for doc_dict in train_features for feature in doc_dict
    )
    valid_features = {
        feature
        for feature, count in features_counter.items()
        if count >= min_occurrences
    }
    filtered_train_features = [
        {
            feature: value
            for feature, value in doc_dict.items()
            if feature in valid_features
        }
        for doc_dict in train_features
    ]
    return filtered_train_features


def get_labels(files):
    """ Funzione che estrae le labels dai files"""
    hs = []
    stereotypes = []
    for file_name in files:
        classes = [int(num) for num in re.findall(r"\d+", file_name)]
        hs.append(classes[1])
        stereotypes.append(classes[2])
    return hs, stereotypes


hs, stereotypes = get_labels(train_files)


train = extract_features(train_dataset)
train = filter_features(train, 1)


def run_experiment(features, labels):
    # Imposta lo scaler, scala le features, effettua ricerca degli iper-parametri migliori
    # per il linear SVC. Una volta trovati li utilizza per creare il LinearSVC, lo fitta e printa
    # il report della performance con 5fold crossvalidation, riportando accuracy 
    # media e std
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(features)

    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)
    
    param_grid = {
        'C': [0.1, 1.0, 3.0, 10.0],
        'dual': [True, False]
    }
    
    svc = LinearSVC(max_iter=25000)
    
    grid_search = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, labels)
    
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)
    
    svc = LinearSVC(C=best_params['C'], dual=best_params['dual'], max_iter=25000)
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


print(" PRIMO ESPERIMENTO BASATO SU UNIGRAMMI, BIGRAMMI E TRIGRMMI DI PAROLE, LEMMI E POS")

train = extract_features(train_dataset)
train = filter_features(train, 1)

run_experiment(train, hs)


def extract_features(dataset):
    dataset_ft = []
    for document in dataset:
        unigrams_word = normalize(extract_ngrams(document, 'word', 1))
        bigramns_word = normalize(extract_ngrams(document,'word', 2))
        trigrams_word = normalize(extract_ngrams(document, 'word', 3))
        unigrams_pos = normalize(extract_ngrams(document, 'pos', 1))
        bigrams_pos = normalize(extract_ngrams(document, 'pos', 2))
        trigrams_pos = normalize(extract_ngrams(document, 'pos', 3))
        # fourgrams_pos = normalize(extract_ngrams(document, 'pos', 4))
        # unigrams_lemma = normalize(extract_ngrams(document, "lemma", 1))
        # bigramns_lemma = normalize(extract_ngrams(document, "lemma", 2))
        # trigrams_lemma = normalize(extract_ngrams(document, 'lemma', 3))
        # fourgrams_lemma = normalize(extract_ngrams(document, 'lemma', 4))
        fivegrams_char = normalize(extract_char_ngrams(document, 5))
        dataset_ft.append(unigrams_word | bigramns_word | trigrams_word | unigrams_pos | bigrams_pos | trigrams_pos | fivegrams_char)
    return dataset_ft

train = extract_features(train_dataset)
train = filter_features(train, 2)

print(" SECONDO ESPERIMENTO BASATO SU UNIGRAMMI, BIGRAMMI, TRIGRAMMI DI PAROLE, BIGRAMMI"
      " E TRIGRAMMI DI POS E PENTAGRAMMI DI CARATTERI")
run_experiment(train, hs)


def extract_features(dataset):
    dataset_ft = []
    for document in dataset:
        # unigrams_word = normalize(extract_ngrams(document, 'word', 1))
        # bigramns_word = normalize(extract_ngrams(document,'word', 2))
        # trigrams_word = normalize(extract_ngrams(document, 'word', 3))
        # unigrams_pos = normalize(extract_ngrams(document, 'pos', 1))
        # bigrams_pos = normalize(extract_ngrams(document, 'pos', 2))
        # trigrams_pos = normalize(extract_ngrams(document, 'pos', 3))
        # fourgrams_pos = normalize(extract_ngrams(document, 'pos', 4))
        unigrams_lemma = normalize(extract_ngrams(document, "lemma", 1))
        bigrams_lemma = normalize(extract_ngrams(document, "lemma", 2))
        # trigrams_lemma = normalize(extract_ngrams(document, 'lemma', 3))
        # fourgrams_lemma = normalize(extract_ngrams(document, 'lemma', 4))
        fivegrams_char = normalize(extract_char_ngrams(document, 5))
        dataset_ft.append(unigrams_lemma | bigrams_lemma | fivegrams_char)
    return dataset_ft

train = extract_features(train_dataset)
train = filter_features(train, 1)

print(" TERZO ESPERIMENTO BASATO SU UNIGRAMMI E BIGRAMMI DI LEMMI, PENTA(?)GRAMMI DI CARATTERI")
run_experiment(train, hs)


def extract_features(dataset):
    dataset_ft = []
    for document in dataset:
        # unigrams_word = normalize(extract_ngrams(document, 'word', 1))
        # bigramns_word = normalize(extract_ngrams(document,'word', 2))
        # trigrams_word = normalize(extract_ngrams(document, 'word', 3))
        # unigrams_pos = normalize(extract_ngrams(document, 'pos', 1))
        bigrams_pos = normalize(extract_ngrams(document, 'pos', 2))
        trigrams_pos = normalize(extract_ngrams(document, 'pos', 3))
        # fourgrams_pos = normalize(extract_ngrams(document, 'pos', 4))
        unigrams_lemma = normalize(extract_ngrams(document, "lemma", 1))
        bigrams_lemma = normalize(extract_ngrams(document, "lemma", 2))
        # trigrams_lemma = normalize(extract_ngrams(document, 'lemma', 3))
        # fourgrams_lemma = normalize(extract_ngrams(document, 'lemma', 4))
        fivegrams_char = normalize(extract_char_ngrams(document, 5))
        dataset_ft.append(bigrams_pos | trigrams_pos | unigrams_lemma | bigrams_lemma | fivegrams_char)
    return dataset_ft

train = extract_features(train_dataset)
train = filter_features(train, 1)
print(" QUARTO ESPERIMENTO BASATO SU BIGRAMMI E TRIGRAMMI DI POS, UNIGRAMMI E BIGRAMMI DI LEMMI, PENTA(?)GRAMMI DI CARATTERI")
run_experiment(train, hs)


def extract_features(dataset):
    dataset_ft = []
    for document in dataset:
        unigrams_word = normalize(extract_ngrams(document, 'word', 1))
        bigramns_word = normalize(extract_ngrams(document,'word', 2))
        trigrams_word = normalize(extract_ngrams(document, 'word', 3))
        unigrams_pos = normalize(extract_ngrams(document, 'pos', 1))
        bigrams_pos = normalize(extract_ngrams(document, 'pos', 2))
        trigrams_pos = normalize(extract_ngrams(document, 'pos', 3))
        # fourgrams_pos = normalize(extract_ngrams(document, 'pos', 4))
        # unigrams_lemma = normalize(extract_ngrams(document, "lemma", 1))
        # bigramns_lemma = normalize(extract_ngrams(document, "lemma", 2))
        # trigrams_lemma = normalize(extract_ngrams(document, 'lemma', 3))
        # fourgrams_lemma = normalize(extract_ngrams(document, 'lemma', 4))
        fivegrams_char = normalize(extract_char_ngrams(document, 5))
        eightgram_char = normalize(extract_char_ngrams(document, 8))
        dataset_ft.append(unigrams_word | bigramns_word | trigrams_word | unigrams_pos | bigrams_pos | trigrams_pos | fivegrams_char | eightgram_char)
    return dataset_ft


train = extract_features(train_dataset)
train = filter_features(train, 1)

print(" QUINTO ESPERIMENTO BASATO SU UNIGRAMMI, BIGRAMMI, TRIGRAMMI DI PAROLE, BIGRAMMI"
      " E TRIGRAMMI DI POS E PENTAGRAMMI DI CARATTERI")
run_experiment(train, hs)


def extract_features(dataset):
    dataset_ft = []
    for document in dataset:
        unigrams_word = normalize(extract_ngrams(document, 'word', 1))
        bigramns_word = normalize(extract_ngrams(document,'word', 2))
        trigrams_word = normalize(extract_ngrams(document, 'word', 3))
        fourgrams_word = normalize(extract_ngrams(document, 'word', 4))
        #unigrams_pos = normalize(extract_ngrams(document, 'pos', 1))
        bigrams_pos = normalize(extract_ngrams(document, 'pos', 2))
        trigrams_pos = normalize(extract_ngrams(document, 'pos', 3))
        fourgrams_pos = normalize(extract_ngrams(document, 'pos', 4))
        # unigrams_lemma = normalize(extract_ngrams(document, "lemma", 1))
        # bigramns_lemma = normalize(extract_ngrams(document, "lemma", 2))
        # trigrams_lemma = normalize(extract_ngrams(document, 'lemma', 3))
        # fourgrams_lemma = normalize(extract_ngrams(document, 'lemma', 4))
        fivegrams_char = normalize(extract_char_ngrams(document, 5))
        eightgram_char = normalize(extract_char_ngrams(document, 8))
        dataset_ft.append(unigrams_word | bigramns_word | trigrams_word | fourgrams_word | bigrams_pos | trigrams_pos | fourgrams_pos |fivegrams_char | eightgram_char)
    return dataset_ft


train = extract_features(train_dataset)
train = filter_features(train, 1)

print(" SESTO ESPERIMENTO BASATO SU UNIGRAMMI, BIGRAMMI, TRIGRAMMI, TETRAGRAMMI DI PAROLE, BIGRAMMI"
      " TRIGRAMMI E TETRAGRAMMMI DI POS E PENTAGRAMMI E 8GRAMMI DI CARATTERI")
run_experiment(train, hs)



def extract_features(dataset):
    dataset_ft = []
    for document in dataset:
        unigrams_word = normalize(extract_ngrams(document, 'word', 1))
        bigramns_word = normalize(extract_ngrams(document,'word', 2))
        # trigrams_word = normalize(extract_ngrams(document, 'word', 3))
        # fourgrams_word = normalize(extract_ngrams(document, 'word', 4))
        #unigrams_pos = normalize(extract_ngrams(document, 'pos', 1))
        bigrams_pos = normalize(extract_ngrams(document, 'pos', 2))
        trigrams_pos = normalize(extract_ngrams(document, 'pos', 3))
        fourgrams_pos = normalize(extract_ngrams(document, 'pos', 4))
        # unigrams_lemma = normalize(extract_ngrams(document, "lemma", 1))
        # bigramns_lemma = normalize(extract_ngrams(document, "lemma", 2))
        trigrams_lemma = normalize(extract_ngrams(document, 'lemma', 3))
        # fourgrams_lemma = normalize(extract_ngrams(document, 'lemma', 4))
        fivegrams_char = normalize(extract_char_ngrams(document, 5))
        eightgram_char = normalize(extract_char_ngrams(document, 8))
        dataset_ft.append(unigrams_word | bigramns_word | trigrams_lemma | bigrams_pos | trigrams_pos | fourgrams_pos |fivegrams_char | eightgram_char)
    return dataset_ft


train = extract_features(train_dataset)
train = filter_features(train, 1)

print(" SETTIMO ESPERIMENTO BASATO SU UNIGRAMMI, BIGRAMMI DI PAROLE, BIGRAMMI"
      " TRIGRAMMI E TETRAGRAMMMI DI POS, TRIGRAMMI DI LEMMA E PENTAGRAMMI E 8GRAMMI DI CARATTERI")
run_experiment(train, hs)


def extract_features(dataset):
    dataset_ft = []
    for document in dataset:
        # unigrams_word = normalize(extract_ngrams(document, 'word', 1))
        # bigramns_word = normalize(extract_ngrams(document,'word', 2))
        # trigrams_word = normalize(extract_ngrams(document, 'word', 3))
        # fourgrams_word = normalize(extract_ngrams(document, 'word', 4))
        #unigrams_pos = normalize(extract_ngrams(document, 'pos', 1))
        # bigrams_pos = normalize(extract_ngrams(document, 'pos', 2))
        #trigrams_pos = normalize(extract_ngrams(document, 'pos', 3))
        #fourgrams_pos = normalize(extract_ngrams(document, 'pos', 4))
        unigrams_lemma = normalize(extract_ngrams(document, "lemma", 1))
        bigramns_lemma = normalize(extract_ngrams(document, "lemma", 2))
        trigrams_lemma = normalize(extract_ngrams(document, 'lemma', 3))
        # fourgrams_lemma = normalize(extract_ngrams(document, 'lemma', 4))
        fivegrams_char = normalize(extract_char_ngrams(document, 5))
        # eightgram_char = normalize(extract_char_ngrams(document, 8))
        dataset_ft.append(unigrams_lemma | bigramns_lemma | trigrams_lemma |fivegrams_char )
    return dataset_ft


train = extract_features(train_dataset)
train = filter_features(train, 1)

print(" OTTAVO ESPERIMENTO BASATO SU UNIGRAMMI, BIGRAMMI, TRIGRAMMI DI LEMMI, PENTAGRAMMI DI CARATTERI")
run_experiment(train, hs)


### Creo il LinearSVM sulla base dei risultati degli esperimenti

train = extract_features(train_dataset)
train = filter_features(train, 1)
vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(train)
scaler = MaxAbsScaler()
X_train = scaler.fit_transform(X_train)
best_params = {
    'C': 0.1,
    'dual': True
}

svc = LinearSVC(C=best_params['C'], dual=best_params['dual'], max_iter=25000)
svc.fit(X_train, hs)


print("Risultati sul test set con il miglior LinearSVM realizzato (HATE SPEECH)")
test_dataset = [get_sentences_from_file(doc_path) for doc_path in test_files]
test = extract_features(test_dataset)
X_test = vectorizer.transform(test)
X_test = scaler.transform(X_test)
test_hs, test_stereotypes = get_labels(test_files)
test_predictions = svc.predict(X_test)
print(classification_report(test_hs, test_predictions, zero_division=0))
ConfusionMatrixDisplay.from_predictions(test_hs, test_predictions, xticks_rotation='vertical', cmap='Blues')
plt.clf()

"""#### Features più importanti"""
svc.coef_
features_names = feature_names = vectorizer.get_feature_names_out()

def f_importances(coef, names, top_n=15):
    
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    imp = imp[-top_n:]  # Select the top 15 most important features
    names = names[-top_n:]  # Corresponding feature names
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

f_importances(svc.coef_[0], features_names, 15)

