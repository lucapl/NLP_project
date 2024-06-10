import string

import transformers


class Summarizer:
    def __init__(
        self,
        pipeline: transformers.TextGenerationPipeline,
        prompt_template: string.Template = string.Template("$dialogue"),
    ):
        self.pipeline = pipeline
        self.prompt_template = prompt_template
        assert prompt_template.get_identifiers() == ["dialogue"], "Invalid template"

    def __call__(self, dialogues: list[str]) -> list[str]:
        prompts = []
        for dial in dialogues:
            prompts.append(self.prompt_template.substitute(dialogue=dial))

        predictions = self.pipeline(prompts)
        assert predictions is not None
        summaries = []
        for pred in predictions:
            assert isinstance(pred, list)
            summaries.append(pred[0]["generated_text"])
        return summaries
