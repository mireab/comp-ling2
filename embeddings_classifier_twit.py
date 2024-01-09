# Importo le librerie necessarie
import math
import os
import shutil
import zipfile
from collections import Counter
import nltk
import numpy as np
import pandas as pd
import regex as re
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.svm import LinearSVC

# Estraggo i file dal file ZIP nella directory di destinazione
zip_file_path = "Processing UD/9437.zip"
destination_dir = "annotated_docs"

with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(destination_dir)

# Flattening della cartella
source_dir = "annotated_docs"
destination_dir = "annotated_docs"
os.makedirs(destination_dir, exist_ok=True)
for root, _, files in os.walk(source_dir):
    for file in files:
        source_file_path = os.path.join(root, file)
        shutil.move(source_file_path, os.path.join(destination_dir, file))
# Ottengo la lista di tutti i files con i documenti di training e testing
file_list = os.listdir("annotated_docs")
# Inizializzo le liste e li divido in train e test
train_files = []
test_files = []

for file_name in file_list:
    file_path = os.path.join(source_dir, file_name)
    if "train" in file_name:
        train_files.append(file_path)
    elif "test" in file_name:
        test_files.append(file_path)


def get_sentences_from_file(src_path):
    """ Estrae le frasi dal file di input estraendone, per ogni parola, 'word', 'lemma' e 'pos' """
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

# Costruisco dataset di addestramento
train_dataset = [get_sentences_from_file(doc_path) for doc_path in train_files]


# Funzioni per formattare il testo seconod le indicazioni fornite per l'utilizzo
# degli embeddings twitter128 e itwac128
def get_digits(text):
    try:
        val = int(text)
    except:
        text = re.sub("\d", "@Dg", text)
        return text
    if val >= 0 and val < 2100:
        return str(val)
    else:
        return "DIGLEN_" + str(len(str(val)))


def normalize_text(word):
    if "URL" in word:
        word = "___URL___"
        return word
    if len(word) > 26:
        return "__LONG-LONG__"
    new_word = get_digits(word)
    if new_word != word:
        word = new_word
    if word[0].isupper():
        word = word.capitalize()
    else:
        word = word.lower()
    return word


def preprocessing(token):
    token = get_digits(token)
    token = normalize_text(token)
    if token == "“":
        token = '"'
    return token


def get_standard_tokens(conllu_document, selected_pos=[]):
    """  Questa funzione consente di ottenere i tokens da rappresentare successivamente, oppure, se è fornito il paramentro pos, consente di selezionarli in base alla loro pos """
    doc_words = []
    for sentence in conllu_document:
        for token in sentence:
            word_to_process = ""
            if selected_pos == []:
                word_to_process = token["word"]
            else:
                if token["pos"] in selected_pos:
                    word_to_process = token["word"]
            if word_to_process != "":
                doc_words.append(preprocessing(word_to_process))
    return doc_words

# Creo la lista (nested )contentente tutti i tokens di tutte le frasi di tutti i documenti
all_doc_tokens = [get_standard_tokens(doc) for doc in train_dataset]


def get_embeddings_from_file(file):
    """  Estrae tutte le rappresentazioni dal file contenente gli embeddings """
    embeddings = {}
    with open(file, "r") as in_file:
        lines = in_file.readlines()
        for line in lines:
            line = line.strip().split("\t")
            embeddings[line[0]] = np.asarray([float(comp) for comp in line[1:]])
    return embeddings


embeddings = get_embeddings_from_file("tweet_embeggings.txt")

# Funzione per rimuovere # da hashtags
def fix_bad_tokens(tokenized_doc):
   fixed_words = []
   for word in tokenized_doc:
       if word[0] == "#" and word != "#" :
           word = word[1:]
           fixed_words.append(word)
       else:
           fixed_words.append(word)
   return fixed_words

all_doc_tokens = [fix_bad_tokens(words) for words in all_doc_tokens]

def compute_embeddings_mean(doc_embeddings):
    """  Calcola la media degli embeddings """
    sum_array = np.sum(doc_embeddings, axis=0)
    mean_array = np.divide(sum_array, len(doc_embeddings))
    return mean_array


def get_embeddings(doc_tokens):
    """  Estrae gli embeddings a partire dai token dei documenti """
    if len(doc_tokens) == 0:
        doc_embeddings = [np.zeros(128)]
    else:
        doc_embeddings = [embeddings[t] for t in doc_tokens if t in embeddings]
    if len(doc_embeddings) == 0:
        doc_embeddings = [np.zeros(128)]
    return doc_embeddings

# Ottengo tutti gli embeddings di tutti i documenti di training
all_doc_embeddings = [get_embeddings(doc) for doc in all_doc_tokens]


def extract_features(dataset):
    """ Funzione per l'estrazione delle features, modificata durante gli esperimenti per modificare le pos da tenere in considerazione """
    pos_filtered_tokens = [
        [
            get_standard_tokens(k, selected_pos="ADJ"),
            get_standard_tokens(k, selected_pos="NOUN"),
            get_standard_tokens(k, selected_pos="VERB"),
            get_standard_tokens(k, selected_pos="PROPN"),
        ]
        for k in dataset
    ]
    # Lista con embs di token separati di ADJ, NOUN, Verbs
    pos_filtered_embs = [
        [get_embeddings(cat) for cat in doc] for doc in pos_filtered_tokens
    ]
    # Lista con le medie degli embeddings separati
    pos_filtered_separated_means = [
        [compute_embeddings_mean(cat) for cat in doc] for doc in pos_filtered_embs
    ]
    train_features = [np.concatenate(doc) for doc in pos_filtered_separated_means]
    return train_features


def get_labels(files):
    """ Estrae labels di hs e stereotypes """
    hs = []
    stereotypes = []
    for file_name in files:
        classes = [int(num) for num in re.findall(r"\d+", file_name)]
        hs.append(classes[1])
        stereotypes.append(classes[2])
    return hs, stereotypes

# Estraggo le labels del training set
hs, stereotypes = get_labels(train_files)


def run_experiment(features, labels):
    """ Imposta lo scaler, scala le features, effettua ricerca degli iper-parametri migliori
    per il linear SVC. Una volta trovati li utilizza per creare il LinearSVC, lo fitta e printa
    il report della performance con 5fold crossvalidation, riportando accuracy 
    media e std """
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(features)
    
    param_grid = {
        'C': [0.1, 1.0, 3.0, 10.0],
        'dual': [True, False]
    }
    
    svc = LinearSVC(max_iter=15000)
    
    grid_search = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, labels)
    
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)
    
    svc = LinearSVC(C=best_params['C'], dual=best_params['dual'], max_iter=15000)
    svc.fit(X_train, hs)
    
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


# Conduco una serie di esperimenti, poi di volta in volta modifico le features estratte
print("PRIMO MODELL0 - BASATO SU PAROLE PIENE E MEDIA SEPARATA")
train_features = extract_features(train_dataset)
run_experiment(train_features, hs)

print("SECONDO MODELLO - BASATO SULLA MEDIA NON SEPARATA")
all_doc_means = [compute_embeddings_mean(doc) for doc in all_doc_embeddings]
run_experiment(all_doc_means, hs)

# Provo a usare tfidf per pesare embeddings
def compute_tf_idf(tokenized_documents):
    """ Funzione che calcola tfidf da usare come pesi nel calcolo di una media ponderata"""
    tf = [Counter(doc) for doc in tokenized_documents]
    df = {}
    for doc in tokenized_documents:
        unique_words = set(doc)
        for word in unique_words:
            if word not in df:
                df[word] = 1
            else:
                df[word] += 1
    N = len(tokenized_documents)
    idf = {word: math.log(N / df[word]) for word in df}
    tfidf = []
    for doc in tf:
        tfidf_doc = {word: tf_word * idf[word] for word, tf_word in doc.items()}
        tfidf.append(tfidf_doc)
    return tfidf


tfidf = compute_tf_idf(all_doc_tokens)


def get_weighted_embeddings(all_embeddings, all_tokens, tf_idf):
    """ Funzione che ottiene embeddings pesati con tfidf """
    all_docs_weighted_embs = []
    for i in range(len(all_embeddings)):  # all docs level
        doc_weighted_embeddings = []
        for l in range(len(all_embeddings[i])):
            # single level
            token = all_tokens[i][l]
            embedding = all_embeddings[i][l]
            weight = tf_idf[i][token]
            weighted_value = weight * embedding
            doc_weighted_embeddings.append(weighted_value)
        all_docs_weighted_embs.append(doc_weighted_embeddings)
    return all_docs_weighted_embs


all_docs_weighted_embs = get_weighted_embeddings(
    all_doc_embeddings, all_doc_tokens, tfidf
)
print("TERZO MODELLO - BASATO SU EMBEDDINGS PESATI")
documents_weighted_means = [
    compute_embeddings_mean(doc_embs) for doc_embs in all_docs_weighted_embs
]
run_experiment(documents_weighted_means, hs)

print(
    "QUARTO MODELLO - BASATO SULLA SOMMA DI VETTORI NORMALIZZATI CON L2 NORM POS FILTRATI"
)
pos_filtered_tokens = [
    [
        get_standard_tokens(k, selected_pos="ADJ"),
        get_standard_tokens(k, selected_pos="NOUN"),
        get_standard_tokens(k, selected_pos="VERB"),
        get_standard_tokens(k, selected_pos="PROPN"),
    ]
    for k in train_dataset
]
# Lista con embs di token separati di ADJ, NOUN, Verbs
pos_filtered_embs = [
    [get_embeddings(cat) for cat in doc] for doc in pos_filtered_tokens
]
# Lista con le medie degli embeddings separati
# Normalizzo ogni vettore con la norma L2
normalized_embeddings = [
    [
        [
            embedding / np.linalg.norm(embedding)
            if np.linalg.norm(embedding) > 0
            else np.zeros_like(embedding)
            for embedding in cat
        ]
        or [np.zeros_like(cat[0])]  
        for cat in doc
    ]
    for doc in pos_filtered_embs
]  


def compute_embeddings_sum(doc_embeddings):
    """ Calcola somma embeddings """
    return np.sum(doc_embeddings, axis=0)

# Ottengo somme normalizzate
all_docs_normalized_sums = [
    [compute_embeddings_sum(cat) for cat in doc] for doc in normalized_embeddings
]
train_features = [np.concatenate(doc) for doc in all_docs_normalized_sums]

run_experiment(train_features, hs)


print(
    "QUINTO MODELLO - BASATO SULLA SOMMA DI VETTORI NORMALIZZATI CON L2 NORM NON FILTRATI"
)
normalized_embeddings = [
    [embedding / np.linalg.norm(embedding) for embedding in doc]
    for doc in all_doc_embeddings
]
all_docs_normalized_sums = [
    compute_embeddings_sum(doc) for doc in normalized_embeddings
]

run_experiment (all_docs_normalized_sums, hs)


print("SESTO MODELL0 - BASATO SU AGGETTIVI E NOMI E MEDIA SEPARATA")

def extract_features(dataset):
    pos_filtered_tokens = [
        [
            get_standard_tokens(k, selected_pos="ADJ"),
            get_standard_tokens(k, selected_pos="NOUN"),
        ]
        for k in dataset
    ]
    # Lista con embs di token separati di ADJ, NOUN
    pos_filtered_embs = [
        [get_embeddings(cat) for cat in doc] for doc in pos_filtered_tokens
    ]
    # Lista con le medie degli embeddings separati
    pos_filtered_separated_means = [
        [compute_embeddings_mean(cat) for cat in doc] for doc in pos_filtered_embs
    ]
    train_features = [np.concatenate(doc) for doc in pos_filtered_separated_means]
    return train_features


train_features = extract_features(train_dataset)

run_experiment(train_features, hs)

print("SETTIMO MODELL0 - MEDIA DI TOKEN ESCLUSE STOPWORDS")
nltk.download("stopwords")
# Scarico le stopwords fornite da nltk
italian_stopwords = stopwords.words("italian")
# Seleziono solo token non stopword
non_sw_tokens = [
    [token for token in doc if token not in italian_stopwords] for doc in all_doc_tokens
]
# Estraggo embeddings
all_doc_embeddings = [get_embeddings(doc) for doc in non_sw_tokens]
# Calcolo medie
all_doc_means = [compute_embeddings_mean(doc) for doc in all_doc_embeddings]

run_experiment(all_doc_means, hs)

# Risultati del classificatore con miglior accuracy
scaler = MaxAbsScaler()
X_train = scaler.fit_transform(all_doc_means)
svc = LinearSVC(C=1.0, dual=True, max_iter=15000)
svc.fit(X_train, hs)

test_dataset = [get_sentences_from_file(doc_path) for doc_path in test_files]
all_test_doc_tokens = [get_standard_tokens(doc) for doc in test_dataset]
non_sw_test_tokens = [
    [token for token in doc if token not in italian_stopwords] for doc in all_test_doc_tokens
]
all_doc_test_embeddings = [get_embeddings(doc) for doc in non_sw_test_tokens]
all_doc_test_means = [compute_embeddings_mean(doc) for doc in all_doc_test_embeddings]
hs_test, stereotypes_test =  get_labels (test_files)

X_test = scaler.transform(all_doc_test_means)
test_predictions = svc.predict(X_test)
print(classification_report(hs_test, test_predictions,  zero_division=0))


# Risultati con l'altro classificatore con miglior accuracy
scaler = MaxAbsScaler()
X_train = scaler.fit_transform(all_docs_normalized_sums)
svc = LinearSVC(C=10.0, dual=True, max_iter=15000)
svc.fit(X_train, hs)
# Test del modello selazionato
test_dataset = [get_sentences_from_file(doc_path) for doc_path in test_files]
all_test_doc_tokens = [get_standard_tokens(doc) for doc in test_dataset]
all_doc_test_embeddings = [get_embeddings(doc) for doc in all_test_doc_tokens]
normalized_test_embeddings = [
    [embedding / np.linalg.norm(embedding) for embedding in doc]
    for doc in all_doc_test_embeddings
]
all_docs_test_normalized_sums = [
    compute_embeddings_sum(doc) for doc in normalized_test_embeddings
]
hs_test, stereotypes_test =  get_labels (test_files)
X_test = scaler.transform(all_docs_test_normalized_sums)
test_predictions = svc.predict(X_test)
print(classification_report(hs_test, test_predictions,  zero_division=0))