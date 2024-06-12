import string

import datasets

import evaluator

DATASET = "Samsung/samsum"


def main():
    dataset = datasets.load_dataset(DATASET)
    print("Running zero shot prompt experiment")
    prompt = (
        "<|user|>\nBriefly summarize this dialogue: $dialogue <|end|>\n<|assistant|>"
    )
    kwargs = {
        "count": 2,
        "prompt_template": prompt,
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "dataset": DATASET,
        "summarizer_type": "TextGeneration",
        "info": "zero-shot",
    }
    evaluator.main(**kwargs)

    assert isinstance(dataset, datasets.DatasetDict)

    print("Running one shot experiment")
    dataset = dataset.shuffle(seed=42)
    shot = dataset["train"].take(1)
    shot_template = string.Template(
        "<|user|>\nBriefly summarize this dialogue: $dialogue <|end|>"
        "\n<|assistant|>$summary"
    )
    oneshot_prompt = (
        shot_template.substitute(dialogue=shot["dialogue"], summary=shot["summary"])
        + prompt
    )
    kwargs["prompt_template"] = oneshot_prompt
    kwargs["info"] = "oneshot"
    evaluator.main(**kwargs)

    print("Running two shot experiment")
    dataset = dataset.shuffle(seed=42)
    shot = dataset["train"].skip(1).take(1)  # type: ignore
    shot_template = string.Template(
        "<|user|>\nBriefly summarize this dialogue: $dialogue <|end|>\n"
        "<|assistant|>$summary"
    )
    twoshot_prompt = (
        shot_template.substitute(dialogue=shot["dialogue"], summary=shot["summary"])
        + oneshot_prompt
    )
    kwargs["prompt_template"] = twoshot_prompt
    kwargs["info"] = "two shot"
    evaluator.main(**kwargs)


if __name__ == "__main__":
    main()
