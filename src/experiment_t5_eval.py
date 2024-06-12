import datasets
from summarizer import T5Summarizer
from evaluator import evaluate_summarizer, calculate_means
import wandb
from wandb.sdk.data_types.trace_tree import Trace

DATASET = "Samsung/samsum"
COUNT = 100


if __name__ == "__main__":
    ds = datasets.load_dataset(DATASET, trust_remote_code=True)
    assert isinstance(ds, datasets.DatasetDict)
    testset = ds["test"]
    testset = testset.shuffle(seed=42).take(COUNT)

    models = ["google-t5/t5-small", "lucapl/t5-summarizer-samsum"]
    for model in models:
        run = wandb.init(project="NLP_summarization", name=model)

        summarizer = T5Summarizer(model)
        results, metrics = evaluate_summarizer(testset, summarizer)
        metrics["dialogue"] = testset["dialogue"]
        metrics["reference"] = testset["summary"]

        for i in range(COUNT):
            inputs = {"system_prompt": "summarize: ",
                      "query": testset[i]["dialogue"], "reference": testset[i]["summary"]}
            outputs = {}
            outputs["prediction"] = results["prediction"][i]
            for row in metrics:
                outputs[row] = metrics[row][i]

            span = Trace(
                name="summarization",
                kind="llm",  # kind can be "llm", "chain", "agent" or "tool"
                metadata={
                    "model_name": model,
                },
                inputs=inputs,
                outputs=outputs,
            )

            span.log(name="summarization")

        means = calculate_means(metrics)
        run.summary.update(means)
        run.finish()
