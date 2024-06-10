import string

import transformers


class Summarizer:
    prompt_template: string.Template

    def __init__(self, prompt_template: string.Template):
        self.prompt_template = prompt_template

    def __call__(self, dialogues: list[str]) -> list[str]:
        prompts = []
        for dial in dialogues:
            prompts.append(self.prompt_template.substitute(dialogue=dial))

        return self.summarize(prompts)

    def summarize(self, prompts: list[str]) -> list[str]:
        raise NotImplementedError


class T5Summarizer(Summarizer):
    def __init__(self, model="google-t5/t5-small", tokenizer="google-t5/t5-small"):
        self.model = transformers.T5ForConditionalGeneration.from_pretrained(model)
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(tokenizer)
        self.prompt_template = string.Template("summarize: $dialogue")

    def summarize(self, prompts: list[str]) -> list[str]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)

        outputs = self.model.generate(  # type: ignore
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=False,
            max_new_tokens=100
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class TextGenerationSummarizer(Summarizer):
    def __init__(self, model, prompt_template: string.Template):
        self.pipe = transformers.pipeline(
            "text-generation",
            model=model,
            max_new_tokens=100,
            return_full_text=False,
            device=0
        )
        self.prompt_template = prompt_template

    def summarize(self, prompts: list[str]) -> list[str]:
        predictions = self.pipe(prompts)
        assert predictions
        summaries = []
        for pred in predictions:
            summaries.append(pred[0]["generated_text"])  # type: ignore
        return summaries
