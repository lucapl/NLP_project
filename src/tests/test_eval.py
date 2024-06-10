import pytest

from eval import evaluate_summaries


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
