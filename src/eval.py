import datetime
import os
import string
import pathlib
import time

import datasets
import evaluate
import pandas as pd

from summarizer import Summarizer, TextGenerationSummarizer

DATASET = "Samsung/samsum"
MODEL = "microsoft/Phi-3-mini-4k-instruct"
BERTSCORE_MODEL = "microsoft/deberta-v3-small"


def main() -> None:
    dataset = datasets.load_dataset(DATASET, trust_remote_code=True)
    assert isinstance(dataset, datasets.DatasetDict)

    summarizer = TextGenerationSummarizer(MODEL, prompt_template=string.Template(
        "<|user|>\nBriefly summarize this dialogue: $dialogue <|end|>\n<|assistant|>"))
    # summarizer = T5Summarizer(model="lucapl/t5-summarizer-samsum")

    testset = dataset["test"]
    testset = testset.shuffle(seed=42).take(10)  # take only small subset to speed up
    metrics = evaluate_summarizer(testset, summarizer)

    df = pd.DataFrame.from_dict(metrics)
    print("Means of the metrics:")
    print(df.drop(columns=["id", "prediction"]).mean())
    df["prediction"] = (
        df["prediction"].str.replace("\n", " ").str.replace("\r", " ")
    )  # no newlines in csv

    filename = datetime.datetime.now().strftime("%Y_%d_%m_%H_%M") + "_results.csv"

    output_path = pathlib.Path("outputs/")
    if not output_path.exists():
        os.mkdir(output_path)
    df.to_csv(output_path / filename)


def evaluate_summarizer(testset: datasets.Dataset, summarizer: Summarizer) -> dict:
    """Evaluates given summarizer
    Args:
       testset: dataset which contains columns: "dialogue", "summary" and "id"
          where dialogue is a text to summarize and summary is refernce ground truth
       summarizer: Summarizer that on calling summarizes list of dialogues
    Returns:
        dictionary with different metrics calculated for each dialogue in testset
        along withg "id" for identification and "predictions" (generated text)
    """
    dialogues = testset["dialogue"]
    predictions = generate_summaries(dialogues, summarizer)

    references = testset["summary"]
    results = evaluate_summaries(predictions, references)
    results["id"] = testset["id"]
    results["prediction"] = predictions
    return results


def generate_summaries(dialogues: list[str], summarizer: Summarizer) -> list[str]:
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
    summaries = summarizer(dialogues)
    elapsed = time.time() - start
    print(f"Finished summarizing in {elapsed:.2f} seconds")
    return summaries


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
