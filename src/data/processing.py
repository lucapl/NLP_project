import numpy as np
import transformers as trans
import datasets as ds
from datasets import concatenate_datasets


def get_length(
        tokenizer: trans.PreTrainedTokenizer,
        data: ds.Dataset,
        column: str,
        percentile=85) -> int:
    """
    Tokenizes the input and gets the length of selected percentile
    Args:
        tokenizer: pretrained model tokenizer
        data: raw dataset
        column: column to measure,
        percentile: percentile to choose
    Returns:
        length of the selected percentile
    """
    tokenized_inputs = concatenate_datasets([data["train"], data["test"]]).map(
        lambda x: tokenizer(x[column], truncation=True),
        batched=True,
        remove_columns=["dialogue", "summary"]
    )
    input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
    # take 85 percentile of max length for better utilization
    return int(np.percentile(input_lenghts, percentile))


def preprocess_function(sample,
                        tokenizer: trans.PreTrainedTokenizer,
                        max_target_length: int,
                        max_source_length: int,
                        padding="max_length",
                        data_prefix="summarize: ") -> trans.BatchEncoding:
    """
    Preprocessses the dataset and pads it
    Args:
        sample: batch of data
        tokenizer: pretrained model tokenizer
        max_target_length: max number of target tokens
        max_source_length: max number of input tokens
        padding: how to pad the data
        data_prefix: prompt, prefix to add to each data point
    Returns:
        encoded batch of data
    """
    inputs = [data_prefix + item for item in sample["dialogue"]]

    # tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=padding,
        truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["summary"],
                       max_length=max_target_length,
                       padding=padding,
                       truncation=True)

    # If we are padding here,
    # replace all tokenizer.pad_token_id in the labels by -100
    # when we want to ignore padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(lab
              if lab != tokenizer.pad_token_id
              else -100) for lab in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def tokenize_dataset(data: ds.Dataset,
                     tokenizer: trans.PreTrainedTokenizer,
                     verbose=False,
                     data_prefix="summarize: ") -> ds.Dataset:

    max_source_length = get_length(tokenizer, data, "dialogue")
    if verbose:
        print(f"Max source length: {max_source_length}")

    max_target_length = get_length(tokenizer, data, "summary")
    if verbose:
        print(f"Max target length: {max_target_length}")

    tokenized_dataset = data.map(
        lambda sample: preprocess_function(sample,
                                           tokenizer,
                                           max_target_length,
                                           max_source_length,
                                           data_prefix=data_prefix),
        batched=True,
        remove_columns=["dialogue", "summary", "id"])

    return tokenized_dataset
