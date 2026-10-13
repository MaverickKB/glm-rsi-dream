"""Count labeled prediction errors without making semantic decisions."""

from typing import Any


def class_metrics(expected: Any, predicted: Any) -> Any:
    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted lists must have equal lengths")
    labels = set(expected)
    for value in predicted:
        if value is not None:
            labels.add(value)
    results = {}
    for label in sorted(labels):
        true_positive = 0
        false_positive = 0
        false_negative = 0
        for index, actual in enumerate(expected):
            prediction = predicted[index]
            if actual == label and prediction == label:
                true_positive += 1
            elif prediction == label:
                false_positive += 1
            elif actual == label:
                false_negative += 1
        precision = None
        recall = None
        if true_positive + false_positive:
            precision = true_positive / (true_positive + false_positive)
        if true_positive + false_negative:
            recall = true_positive / (true_positive + false_negative)
        results[label] = {"tp": true_positive, "fp": false_positive, "fn": false_negative,
                          "precision": precision, "recall": recall}
    return results
