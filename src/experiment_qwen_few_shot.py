import datasets
from summarizer import TextGenerationSummarizer
from evaluator import evaluate_summarizer, calculate_means
import wandb
import random
import string
from wandb.sdk.data_types.trace_tree import Trace

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

    model = "Qwen/Qwen2-0.5B-Instruct"
    template = f"{USER_INST}$dialogue{AGENT_INST}"
    summarizer = TextGenerationSummarizer(model, string.Template(template))

    random.seed(42)
    run = wandb.init(project="NLP_summarization", name=model)

    for p in range(3):
        trainset = trainset.shuffle(seed=42)
        number_of_shots = random.randint(1, 5)
        shots = trainset.take(number_of_shots)
        prompt = ""
        for shot in shots:
            prompt += USER_INST + shot["dialogue"]  # type: ignore
            prompt += AGENT_INST + shot["summary"] + "\n"  # type: ignore

        new_template = string.Template(prompt + template)

        results, metrics = evaluate_summarizer(testset, summarizer)
        metrics["dialogue"] = testset["dialogue"]
        metrics["reference"] = testset["summary"]

        run.log({"prompt": prompt})

        for i in range(COUNT):
            inputs = {"query": testset[i]["dialogue"],
                      "reference": testset[i]["summary"]}

            outputs = {}
            outputs["prediction"] = results["prediction"][i]
            for row in metrics:
                outputs[row] = metrics[row][i]

            span = Trace(
                name=str(p),
                kind="llm",  # kind can be "llm", "chain", "agent" or "tool"
                metadata={
                    "model_name": model,
                },
                inputs=inputs,
                outputs=outputs,
            )

            span.log(name=str(p))

        means = calculate_means(metrics)
        run.log()
        run.summary.update(means)
