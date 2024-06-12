import string

import datasets
import wandb
from wandb.sdk.data_types.trace_tree import Trace

from evaluator import calculate_means, evaluate_summarizer
from summarizer import TextGenerationSummarizer

DATASET = "Samsung/samsum"
COUNT = 100

USER_INST = "[summarize the following dialogue]: "
AGENT_INST = "[end]\n[summary]: "

if __name__ == "__main__":
    ds = datasets.load_dataset(DATASET, trust_remote_code=True)
    assert isinstance(ds, datasets.DatasetDict)
    trainset = ds["train"]
    testset = ds["test"]
    testset = testset.shuffle(seed=42).take(COUNT)

    model = "Qwen/Qwen2-1.5B-Instruct"

    prompts_to_check = [
        "Dialogue: $dialogue\nSummarize in one sentence: ",
        "Dialogue: $dialogue\nTL;DR:",
        "TL;DR: $dialogue\n",
    ]

    for i, prompt in enumerate(prompts_to_check):
        summarizer = TextGenerationSummarizer(model, string.Template(prompt))

        run = wandb.init(project="NLP_summarization", name=model + str(i))

        results, metrics = evaluate_summarizer(testset, summarizer)
        metrics["dialogue"] = testset["dialogue"]
        metrics["reference"] = testset["summary"]

        for i in range(COUNT):
            inputs = {
                "query": testset[i]["dialogue"],
                "reference": testset[i]["summary"],
                "prompt": prompt,
            }

            outputs = {}
            outputs["prediction"] = results["prediction"][i]
            for row in metrics:
                outputs[row] = metrics[row][i]

            span = Trace(
                name="summarize",
                kind="llm",  # kind can be "llm", "chain", "agent" or "tool"
                metadata={
                    "model_name": model,
                },
                inputs=inputs,
                outputs=outputs,
            )

            span.log(name=str(i))

        means = calculate_means(metrics)
        run.summary.update(means)
        run.finish()
