from pathlib import Path

from decision_lab.data.validate import read_examples, validate_dataset


def make_lexical_estimator(seed: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 1)), LogisticRegression(random_state=seed, max_iter=1000)
    )


def lexical_baseline(directory: Path, seed: int) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

    validate_dataset(directory)
    train = read_examples(directory / "train.jsonl")
    development = read_examples(directory / "development.jsonl")
    estimator = make_lexical_estimator(seed)
    estimator.fit([row.prompt for row in train], [row.label.value for row in train])
    labels = [row.label.value for row in development]
    predictions = estimator.predict([row.prompt for row in development])
    probabilities = estimator.predict_proba([row.prompt for row in development])
    return {
        "split": "development",
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "nll": log_loss(labels, probabilities, labels=estimator.classes_),
        "classes": estimator.classes_.tolist(),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=estimator.classes_
        ).tolist(),
        "training_questions": len(train),
        "evaluation_questions": len(development),
    }
