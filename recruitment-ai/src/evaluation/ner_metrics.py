from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

def evaluate_ner(y_true: list[list[str]], y_pred: list[list[str]]) -> dict:
    """
    Evaluate NER predictions using seqeval.
    y_true: list of true labels for each sentence (e.g., [['B-SKILL', 'I-SKILL', 'O'], ...])
    y_pred: list of predicted labels
    """
    metrics = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred)
    }
    return metrics

def print_ner_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    print("NER Evaluation Report:")
    print(report)
