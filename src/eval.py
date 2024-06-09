import datetime
import time

import datasets
import evaluate
import pandas as pd
import transformers

from src.data.processing import tokenize_dataset

DATASET = "Samsung/samsum"
MODEL = "google-t5/t5-small"
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"


def main() -> None:
    dataset = datasets.load_dataset(DATASET, trust_remote_code=True)
    assert isinstance(dataset, datasets.DatasetDict)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    # pipe = transformers.pipeline("summarization", model=MODEL)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=".output/t5-summarizer",
        learning_rate=1e-3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False
    )

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        model,
        return_tensors="pt",
        label_pad_token_id=-100,
        pad_to_multiple_of=8)

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator
    )

    dialogues = dataset["test"]["dialogue"]
    print(f"Summarizing {len(dialogues)} docs")
    start = time.time()
    # summaries = pipe(dialogues, max_length=100)
    preds = trainer.predict(tokenized_dataset["test"])
    elapsed = time.time() - start
    print(f"Finished summarizing in {elapsed:.2f} seconds")

    predictions = preds[""]
    # assert summaries is not None
    # for summary in summaries:
    #     assert isinstance(summary, dict)
    #     predictions.append(summary["summary_text"])

    references = dataset["test"]["summary"]

    metrics = evaluate_summaries(predictions, references)

    df = pd.DataFrame.from_dict(metrics)
    print("Means of the metrics:")
    print(df.mean())

    df["samsum_test_id"] = dataset["test"]["id"]
    filename = datetime.datetime.now().strftime("%Y_%d_%m_%H_%M") + "_results.csv"

    df.to_csv(filename)


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
