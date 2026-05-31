from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset, Dataset
import numpy as np
import seqeval.metrics
import json
import os

def load_data(file_path):
    # Load dataset
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def train_phobert_ner():
    model_name = "xlm-roberta-base"
    train_file = "data/annotated/train.jsonl"
    
    if not os.path.exists(train_file):
        print(f"Không tìm thấy file {train_file}. Hãy chạy scripts.auto_annotate_ner trước!")
        return

    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    raw_data = load_data(train_file)

    # 1. Trích xuất tất cả các nhãn (Labels) có trong dataset
    unique_labels = set()
    for item in raw_data:
        unique_labels.update(item["ner_tags"])
    
    # Đảm bảo có nhãn 'O'
    unique_labels.add("O")
    label_list = sorted(list(unique_labels))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}

    # 2. SPLIT TRƯỚC KHI AUGMENT để tránh data leak
    #    Chia 80% train, 20% test trên dữ liệu gốc (chưa nhân bản)
    dataset = Dataset.from_list(raw_data)
    split_dataset_raw = dataset.train_test_split(test_size=0.2, seed=42)

    # 3. Augment CHỈ trên tập train (nhân bản để tăng số lượng mẫu huấn luyện)
    TARGET_TRAIN_COUNT = 240  # Mục tiêu ~240 mẫu train (tương ứng 300 tổng với 20% test)
    train_data = list(split_dataset_raw["train"])
    train_count = len(train_data)
    if train_count > 0 and train_count < TARGET_TRAIN_COUNT:
        multiplier = TARGET_TRAIN_COUNT // train_count
        remainder = TARGET_TRAIN_COUNT % train_count
        train_data = (train_data * multiplier) + train_data[:remainder]
    
    train_dataset = Dataset.from_list(train_data)
    test_dataset = split_dataset_raw["test"]

    print(f"  Train samples (sau augment): {len(train_dataset)}")
    print(f"  Test samples (gốc, không augment): {len(test_dataset)}")
    
    # 4. Căn lề tokenization cho sub-words (HuggingFace requirement)
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=256)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100) # Ignore special tokens
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id.get(label[word_idx], -100))
                else:
                    label_ids.append(-100) # Ignore subwords
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
    
    training_args = TrainingArguments(
        output_dir="experiments/phobert-ner",
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("Bat dau qua trinh huan luyen (Training)...")
    trainer.train()
    
    # Lưu mô hình cuối cùng
    try:
        trainer.save_model("experiments/phobert-ner-final")
        print("Huan luyen hoan tat! Mo hinh da duoc luu tai thu muc: experiments/phobert-ner-final")
    except Exception as e:
        print(f"Loi khi luu vao thu muc chinh (Streamlit dang lock model): {e}")
        fallback_dir = "experiments/phobert-ner-final-new"
        print(f"Dang luu thu vao thu muc thay the: {fallback_dir}")
        trainer.save_model(fallback_dir)
        print(f"Huan luyen hoan tat! Mo hinh da duoc luu tai thu muc: {fallback_dir}")

if __name__ == "__main__":
    train_phobert_ner()
