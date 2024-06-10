import datetime
import time

import datasets
import evaluate
import pandas as pd
import transformers

DATASET = "Samsung/samsum"
MODEL = "google-t5/t5-small"
BERTSCORE_MODEL = "microsoft/deberta-v3-small"


def main() -> None:
    dataset = datasets.load_dataset(DATASET, trust_remote_code=True)
    assert isinstance(dataset, datasets.DatasetDict)

    pipe = transformers.pipeline("text-generation", model=MODEL)
    testset = dataset['test'].take(1)
    metrics = evaluate_pipeline(testset, pipe)

    df = pd.DataFrame.from_dict(metrics)
    print("Means of the metrics:")
    print(df.drop(columns="id").mean())

    filename = datetime.datetime.now().strftime("%Y_%d_%m_%H_%M") + "_results.csv"
    df.to_csv(filename)


def evaluate_pipeline(testset: datasets.Dataset, pipe: transformers.Pipeline) -> dict:
    dialogues = testset["dialogue"]
    predictions = generate_summaries(dialogues, pipe)

    references = testset["summary"]
    metrics = evaluate_summaries(predictions, references)
    metrics["id"] = testset["id"]
    return metrics


def generate_summaries(dialogues: list[str], pipe: transformers.Pipeline) -> list[str]:
    """Generate summaries of dialogues using huggingface transformers pipeline
        Args:
            dialogues: list of dialogues to summarize
            pipeline: pipeline that generates summaries for dialogues
              resulting summary should be in its dict result in field 'generated_text'
        Returns:
            list of predicted summaries
    """
    print(f"Summarizing {len(dialogues)} docs")
    start = time.time()
    summaries = pipe(dialogues)
    elapsed = time.time() - start
    print(f"Finished summarizing in {elapsed:.2f} seconds")

    predictions = []
    assert summaries is not None
    for summary in summaries:
        assert isinstance(summary, list)
        predictions.append(summary[0]["generated_text"])
    return predictions


def evaluate_summaries(predictions: list[str], references: list[str]) -> dict:
    """
    Evaluate summaries of text using BERTscore and ROUGE metrics
    Args:
        predictions: list of generated summaries
        references: list of reference (ground truth) summaries
    Returns:
        dict with keys and values corresponding to evaluated metrics:
            - keys for BERTscore: precision, recall and f1
            - keys for ROUGE: rouge1, rouge2, rougeL, rougeLsum
    """
    bertscore = evaluate.load("bertscore")

    print(f"Evaluating predictions using BERTscore ({BERTSCORE_MODEL})")
    start = time.time()
    bert_results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type=BERTSCORE_MODEL,
    )
    elapsed = time.time() - start
    print(f"Evaluated using BERTscore in {elapsed:.2f} seconds")
    assert bert_results is not None
    del bert_results["hashcode"]

    rouge = evaluate.load("rouge")
    print("Evalyuating predictions using rouge")
    start = time.time()
    rouge_results = rouge.compute(
        predictions=predictions, references=references, use_aggregator=False
    )
    elapsed = time.time() - start
    print(f"Evaluated in {elapsed:.2f} seconds")
    assert rouge_results is not None

    bert_results.update(rouge_results)
    return bert_results


if __name__ == "__main__":
    main()
