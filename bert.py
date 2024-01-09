# Importo le librerie necessarie
import csv
import pandas as pd
import evaluate
import datasets
from matplotlib import pyplot as plt
import transformers
import sklearn
import numpy as np
import accelerate
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# Imposto i path ai file di train e test
file = 'haspeede2_dev/haspeede2_train_taskAB.tsv'
file_test = 'haspeede2_dev/haspeede2_test_taskAB-tweets.tsv'


def create_df_from_file(file_path):
    """ Funzione che crea un dataset partendo dal file.tsv, estraendo id, testo e label """
    data = []
    with open(file_path, 'r', encoding='utf-8', newline='') as infile:
        lines = infile.readlines()
        for line in lines[1:]:
            row = line.strip().split('\t')
            ident = row[0]
            testo = row[1]
            hs = row[2]
            row_dict = {'id': ident, 'text': testo, 'label': hs}
            data.append(row_dict)
    df = pd.DataFrame(data)
    return df

# Creo df di train (e poi lo splitto) e df di test
train_df = create_df_from_file(file)
test_df = create_df_from_file(file_test)
train_df = datasets.Dataset.from_pandas(train_df)
test_df = datasets.Dataset.from_pandas(test_df)
train_dev = train_df.train_test_split(test_size=0.1)
train = train_dev["train"]
dev = train_dev["test"]

# Seleziono il modello e importo modello e tokenizer
model_name = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    """ Funzione che tokenizza i testi con bert e associa la relativa label """
    tokens = tokenizer(batch['text'], padding=True, truncation=True, max_length=512)
    tokens['label'] = [int(label) for label in batch["label"]]
    return tokens

# Applico la funzione tokenize a tutti i set
train = train.map (tokenize, batched = True)
test = test_df.map (tokenize, batched = True)
dev = dev.map (tokenize, batched = True)

# Formatto i set per renderli utilizzabili dal modello
train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
dev.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Imposto il numero di epochs per il fine tuning
num_epochs = 5

# Setto gli argomenti di training
training_args = TrainingArguments(
    f"{model_name}-finetuned",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

# Inizializzo liste per storare le losses
train_losses = []
valid_losses = []

def compute_metrics(eval_pred):
    """ Funzione che calcola le metriche """
    f1_metric = evaluate.load("f1")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1_metric.compute(predictions=predictions, references=labels, average="weighted")

# Inizializzo il Trainer
trainer = Trainer(
    model,
    training_args,
    train_dataset=train,
    eval_dataset=dev,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Inizio il training con gli argomenti settati, usando le metriche definite
trainer.train()

# Utilizzo i logs per registare le loss di ogni epoch
for logs in trainer.state.log_history:
    if 'loss' in logs.keys():
        train_losses.append(logs['loss'])
    if 'eval_loss' in logs.keys():
        valid_losses.append(logs['eval_loss'])

# Plotto le curve di training e validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True)
plt.show()
plt.clf()
# Salvo il modello
trainer.save_model("./FINETUNED_ALBERTO_MODEL")

# Testo il modello sul test set
output_predictions = trainer.predict(test)
y_test = test["label"].tolist()
y_pred = np.argmax(output_predictions.predictions, axis=1)
report = classification_report(y_test, y_pred)
cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation='vertical', cmap='Blues')

# Stampo i risultati
print("Classification Report:")
print(report)
print()

print("Confusion Matrix:")
print(cm)

