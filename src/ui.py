import gradio as gr
import transformers

models = ["google-t5/t5-small",
          "lucapl/t5-summarizer-samsum"]
# , "google-t5/t5-base", "google-t5/t5-large"]
pipelines: list[transformers.Pipeline] = [None] * 2


def load_pipeline(model) -> transformers.Pipeline:
    global pipelines
    idx = models.index(model)
    if pipelines[idx] is None:
        pipelines[idx] = transformers.pipeline(
            "summarization", model=models[idx], max_new_tokens=100
        )
    res = pipelines[idx]
    assert res is not None
    return res


def summarize(text, model):
    print(model)
    pipe = load_pipeline(model)

    results = pipe(text)
    assert isinstance(results, list)
    summary = results[0]
    assert isinstance(summary, dict)
    return summary["summary_text"]


textbox = gr.Textbox(label="Type text to summarize here:", lines=3)
dropdown = gr.Dropdown(choices=models, label="Model", value=models[0])  # type: ignore

gr.Interface(fn=summarize, inputs=[textbox, dropdown], outputs="text").launch()
