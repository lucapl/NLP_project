from summarizer import Summarizer
import string


def test_summarizer_call():
    template = string.Template("Hello $dialogue")

    class TestSummarizer(Summarizer):
        def summarize(self, prompts: list[str]) -> list[str]:
            assert prompts == ["Hello there", "Hello kenobi"]
            return []

    summarizer = TestSummarizer(template)

    dialogues = ["there", "kenobi"]
    summarizer(dialogues)
