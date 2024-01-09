from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets 
from sklearn.metrics import f1_score
import torch
from transformers import Trainer, TrainingArguments

# Load the ALB3RT0 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
model = AutoModelForSequenceClassification.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0", num_labels=2)

file = 'haspeede2_dev/haspeede2_train_taskAB.tsv'
import csv
import pandas as pd

def create_df_from_file(file_path):
    data = []
    with open(file, 'r', encoding='utf-8', newline='') as infile:
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

train_df = create_df_from_file(file)
train_df = datasets.Dataset.from_pandas(train_df)
train_dev = train_df.train_test_split(test_size=0.1)
train = train_dev["train"]
dev = train_dev["test"]

# Define your training arguments
training_args = TrainingArguments(
    output_dir="./tweet_classification_model",
    evaluation_strategy="steps",  # Evaluate at each step
    eval_steps=500,               # Evaluate every 500 steps
    save_steps=500,               # Save model every 500 steps
    num_train_epochs=5,           # Number of training epochs
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    logging_steps=500,
    save_total_limit=1,           # Only keep the last checkpoint
)

# Define a function to compute the evaluation metric (e.g., accuracy)
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    f1 = f1_score(labels, preds, average="weighted")
    return {"f1_score": f1}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=dev,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save the model after training
trainer.save_model("./Alb3rt0 finetuned")

# # Load the trained model
# model = AutoModelForSequenceClassification.from_pretrained("./tweet_classification_model")

# # Create a DataLoader for the test dataset
# test_dataset = dataset["test"]
# test_texts = test_dataset["text"]
# test_labels = test_dataset["label"]

# # Tokenize the test dataset
# test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt")

# # Create a PyTorch DataLoader
# test_dataset = Dataset.from_dict({key: test_encodings[key] for key in ["input_ids", "attention_mask"]})
# test_dataset = test_dataset.with_format("torch")

# # Evaluate the model on the test set
# results = trainer.predict(test_dataset)

# # Compute and print the evaluation metric (e.g., accuracy)
# accuracy = compute_metrics(results)["accuracy"]
# print(f"Test Accuracy: {accuracy:.2f}")
