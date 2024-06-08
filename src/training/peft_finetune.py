import datasets as ds
from datasets import load_dataset, concatenate_datasets
import peft
import transformers as trans
import numpy as np

MODEL_NAME = "google-t5/t5-small"


def get_length(
        tokenizer: trans.PreTrainedTokenizer,
        data: ds.Dataset,
        column: str,
        percentile=85) -> int:
    tokenized_inputs = concatenate_datasets([data["train"], data["test"]]).map(
        lambda x: tokenizer(x[column], truncation=True),
        batched=True,
        remove_columns=["dialogue", "summary"]
    )
    input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
    # take 85 percentile of max length for better utilization
    return int(np.percentile(input_lenghts, percentile))


def preprocess_function(sample,
                        max_target_length: int,
                        padding="max_length",
                        data_prefix="summarize: ") -> trans.BatchEncoding:
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


if __name__ == "__main__":
    data = load_dataset("Samsung/samsum")

    model = trans.AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    tokenizer = trans.AutoTokenizer.from_pretrained(MODEL_NAME)

    max_source_length = get_length(tokenizer, data, "dialogue")
    print(f"Max source length: {max_source_length}")

    max_target_length = get_length(tokenizer, data, "summary")
    print(f"Max target length: {max_target_length}")

    tokenized_dataset = data.map(
        lambda sample: preprocess_function(sample, max_target_length),
        batched=True,
        remove_columns=["dialogue", "summary", "id"])

    peft_config = peft.LoraConfig(
        task_type=peft.TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1)

    model = peft.get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = trans.Seq2SeqTrainingArguments(
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

    data_collator = trans.DataCollatorForSeq2Seq(
        tokenizer,
        model,
        return_tensors="pt",
        label_pad_token_id=-100,
        pad_to_multiple_of=8)

    trainer = trans.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    peft_model_id = "results"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
