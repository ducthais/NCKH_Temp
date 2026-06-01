import os
import json
import numpy as np
from datasets import Dataset
from seqeval.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

model_path = "experiments/phobert-ner-final"
if not os.path.exists(model_path):
    print("Model not found!")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
label2id = model.config.label2id
id2label = model.config.id2label
label_list = [id2label[i] for i in range(len(id2label))]

file_path = 'data/annotated/train.jsonl'
data = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# Use the same split as training (seed=42, test_size=0.2)
dataset = Dataset.from_list(data)
split = dataset.train_test_split(test_size=0.2, seed=42)
test_data = list(split["test"])
dataset = Dataset.from_list(test_data)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=256)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id.get(label[word_idx], -100))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_test = dataset.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(model=model, data_collator=data_collator)
predictions, labels, _ = trainer.predict(tokenized_test)
predictions = np.argmax(predictions, axis=2)

true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

print("HEADER START")
report = classification_report(true_labels, true_predictions, zero_division=0)
print(report)
print("HEADER END")
