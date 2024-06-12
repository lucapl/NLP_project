import pytest

from evaluator import evaluate_summaries, calculate_means


def test_evaluate_summaries():
    predictions = ["John went to buy coffee"]
    references = predictions.copy()
    metrics = evaluate_summaries(predictions, references)

    assert len(metrics.keys()) == 7

    # All metrics should be equal to 1.0 for an exact match
    # BERTscore
    assert metrics["precision"][0] == pytest.approx(1.0)
    assert metrics["recall"][0] == pytest.approx(1.0)
    assert metrics["f1"][0] == pytest.approx(1.0)

    # ROUGE
    assert metrics["rouge1"][0] == 1.0
    assert metrics["rouge2"][0] == 1.0
    assert metrics["rougeL"][0] == 1.0
    assert metrics["rougeLsum"][0] == 1.0


def test_calculate_means():
    M = ["precision", "recall", "f1", "rouge1", "rouge2", "rougeL", "rougeLsum"]
    A = [1, 1, 1, 1, 1, 1, 1]
    B = [2, 3, 4, 5, 0, 0.5, 1]
    metrics = {m: [a, b] for m, a, b in zip(M, A, B)}

    res = calculate_means(metrics)
    R = [0.5, 2, 2.5, 3, 0.5, 0.75, 1]
    assert {"mean_" + m: r for m, r in zip(M, R)}
