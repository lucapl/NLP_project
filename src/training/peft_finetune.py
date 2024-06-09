from datasets import load_dataset
import peft
import transformers as trans
from src.data.processing import tokenize_dataset

MODEL_NAME = "google-t5/t5-small"


def main():
    data = load_dataset("Samsung/samsum")

    model = trans.AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    tokenizer = trans.AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_dataset = tokenize_dataset(data, tokenizer, True)

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


if __name__ == "__main__":
    main()
