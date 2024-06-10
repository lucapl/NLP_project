from src.eval import evaluate_summaries, generate_summaries
from transformers import TextGenerationPipeline


def test_generate_summaries():
    class UpperPipeline(TextGenerationPipeline):
        def __init__(self):
            pass

        def __call__(self, inputs):
            return [{"generated_text": x.upper()} for x in inputs]

    predictions = generate_summaries(
        ["Hello there", "general, Kenobi!"], UpperPipeline()
    )
    assert predictions == ["HELLO THERE", "GENERAL, KENOBI!"]


def test_evaluate_summaries():
    predictions = ["John went to buy coffee"]
    references = predictions.copy()
    metrics = evaluate_summaries(predictions, references)

    assert len(metrics.keys()) == 7

    # All metrics should be equal to 1.0 for an exact match
    # BERTscore
    assert metrics["precision"][0] == 1.0
    assert metrics["recall"][0] == 1.0
    assert metrics["f1"][0] == 1.0

    # ROUGE
    assert metrics["rouge1"][0] == 1.0
    assert metrics["rouge2"][0] == 1.0
    assert metrics["rougeL"][0] == 1.0
    assert metrics["rougeLsum"][0] == 1.0
