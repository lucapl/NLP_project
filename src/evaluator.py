import argparse
import datetime
import os
import string
import pathlib
import time

import datasets
import evaluate
import pandas as pd

from summarizer import Summarizer, T5Summarizer, TextGenerationSummarizer

BERTSCORE_MODEL = "microsoft/deberta-v3-small"


def main(count: int, prompt_template: str, model: str, dataset: str, summarizer_type: str, info: str) -> None:
    ds = datasets.load_dataset(dataset, trust_remote_code=True)

    if summarizer_type == 'T5':
        summarizer = T5Summarizer(model=model)
    else:
        summarizer = TextGenerationSummarizer(
            model, prompt_template=string.Template(prompt_template))

    assert isinstance(ds, datasets.DatasetDict)
    testset = ds["test"]
    testset = testset.shuffle(seed=42).take(count)  # take only small subset to speed up
    metrics = evaluate_summarizer(testset, summarizer)

    df = pd.DataFrame.from_dict(metrics)
    print("Means of the metrics:")
    print(df.drop(columns=["id", "prediction"]).mean())
    df["prediction"] = (
        df["prediction"].str.replace("\n", " ").str.replace("\r", " ")
    )  # no newlines in csv

    filename = datetime.datetime.now().strftime("%Y_%d_%m_%H_%M") + "_results.csv"
    filename = model.split('/')[-1] + '_' + info + '_' + filename

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--count', type=int, default=10,
        help='How many examples from the dataset to summarize'
    )

    parser.add_argument(
        '--prompt-template', type=str, default='summarize the dialogue: $dialogue',
        help="Prompt template to use with LLM (should have $dialogue placeholder in it)"
    )

    parser.add_argument(
        '--model', type=str, default="lucapl/t5-summarizer-samsum", help="Huggingface model"
    )

    parser.add_argument(
        '--dataset', type=str, default="Samsung/samsum"
    )

    parser.add_argument(
        '--summarizer_type', choices=['T5', 'textGeneration'], default='T5',
        help="What type of summarizer to use"
    )

    parser.add_argument(
        '--info', type=str, default='', help='Additional info to put into log filename'
    )

    args = parser.parse_args()
    main(
        count=args.count,
        prompt_template=args.prompt_template,
        model=args.model,
        dataset=args.dataset,
        summarizer_type=args.summarizer_type,
        info=args.info
    )
