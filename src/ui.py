import gradio as gr
import transformers
from summarizer import Summarizer, T5Summarizer

models = ["google-t5/t5-small", "lucapl/t5-summarizer-samsum"]
pipelines: list[Summarizer | None] = [None, None]


def load_pipeline(model) -> Summarizer:
    global pipelines
    idx = models.index(model)
    if pipelines[idx] is None:
        if "t5" in model:
            pipelines[idx] = T5Summarizer(model)
    res = pipelines[idx]
    assert res is not None
    return res


def summarize(text: str, model: str) -> str:
    summarizer = load_pipeline(model)
    results = summarizer([text])
    return results[0]


textbox = gr.Textbox(label="Type text to summarize here:", lines=3)
dropdown = gr.Dropdown(choices=models, label="Model", value=models[0])  # type: ignore

gr.Interface(fn=summarize, inputs=[textbox, dropdown], outputs="text").launch()
