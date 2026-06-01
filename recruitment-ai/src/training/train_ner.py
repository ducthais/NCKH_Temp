from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification, EarlyStoppingCallback
from datasets import Dataset
import numpy as np
import seqeval.metrics
import json
import os
import random
import torch
from collections import Counter

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def augment_by_entity_sampling(data, target_count=500, seed=42):
    random.seed(seed)
    entity_freq = Counter()
    sample_entities = []
    for item in data:
        entities_in_sample = set()
        for tag in item["ner_tags"]:
            if tag.startswith("B-"):
                entities_in_sample.add(tag[2:])
        sample_entities.append(entities_in_sample)
        for e in entities_in_sample:
            entity_freq[e] += 1
    
    if not entity_freq:
        return data
    
    median_freq = sorted(entity_freq.values())[len(entity_freq) // 2]
    rare_entities = {e for e, c in entity_freq.items() if c < median_freq}
    
    sample_weights = []
    for entities in sample_entities:
        rare_count = len(entities & rare_entities)
        weight = 1.0 + (rare_count * 5.0)  # Boost rare entity samples even more
        sample_weights.append(weight)
    
    total_weight = sum(sample_weights)
    sample_probs = [w / total_weight for w in sample_weights]
    
    augmented = list(data)
    if len(data) < target_count:
        additional_needed = target_count - len(data)
        indices = random.choices(range(len(data)), weights=sample_probs, k=additional_needed)
        for idx in indices:
            augmented.append(data[idx])
    
    random.shuffle(augmented)
    return augmented

class WeightedNERTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, dtype=torch.float32).to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        else:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def compute_class_weights(data, label2id, smoothing=0.5):
    tag_counts = Counter()
    for item in data:
        for tag in item["ner_tags"]:
            if tag in label2id:
                tag_counts[label2id[tag]] += 1
    
    num_labels = len(label2id)
    total = sum(tag_counts.values())
    weights = []
    for i in range(num_labels):
        count = tag_counts.get(i, 1)
        w = (total / (num_labels * count)) ** smoothing
        weights.append(w)
    
    mean_w = sum(weights) / len(weights)
    weights = [w / mean_w for w in weights]
    return weights

def train_phobert_ner():
    model_name = "xlm-roberta-base"
    train_file = "data/annotated/train.jsonl"
    
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    raw_data = load_data(train_file)

    unique_labels = set()
    for item in raw_data:
        unique_labels.update(item["ner_tags"])
    unique_labels.add("O")
    label_list = sorted(list(unique_labels))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}

    dataset = Dataset.from_list(raw_data)
    split_dataset_raw = dataset.train_test_split(test_size=0.2, seed=42)

    train_data_raw = list(split_dataset_raw["train"])
    train_data = augment_by_entity_sampling(train_data_raw, target_count=800, seed=42)
    
    train_dataset = Dataset.from_list(train_data)
    test_dataset = split_dataset_raw["test"]

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

    split_dataset = {
        "train": train_dataset.map(tokenize_and_align_labels, batched=True),
        "test": test_dataset.map(tokenize_and_align_labels, batched=True),
    }
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, 
        num_labels=len(label_list), 
        id2label=id2label, 
        label2id=label2id
    )
    
    class_weights = compute_class_weights(train_data, label2id, smoothing=0.5)
    
    training_args = TrainingArguments(
        output_dir="experiments/phobert-ner",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=35,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        logging_steps=50,
        fp16=False,
        seed=42,
    )
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        return {
            "precision": seqeval.metrics.precision_score(true_labels, true_predictions),
            "recall": seqeval.metrics.recall_score(true_labels, true_predictions),
            "f1": seqeval.metrics.f1_score(true_labels, true_predictions),
            "accuracy": seqeval.metrics.accuracy_score(true_labels, true_predictions),
        }

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = WeightedNERTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=7)],
    )
    
    trainer.train()
    
    try:
        trainer.save_model("experiments/phobert-ner-final")
    except Exception as e:
        trainer.save_model("experiments/phobert-ner-final-new")

if __name__ == "__main__":
    train_phobert_ner()
