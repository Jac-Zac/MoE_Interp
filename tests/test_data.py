"""Tests for dataset prompt loading."""

from datasets import Dataset

from moe_interp.io.data import load_dataset_prompts


class _DummyTokenizer:
    def __init__(self):
        self.chat_calls = []
        self.text_calls = []

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt=False,
        tokenize=False,
        **kwargs,
    ):
        self.chat_calls.append(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                **kwargs,
            }
        )
        return [101, 102, 103]

    def __call__(self, text, add_special_tokens=False, tokenize=False, **kwargs):
        self.text_calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
                "tokenize": tokenize,
            }
        )
        return {"input_ids": [201, 202, 203]}


def test_triviaqa_uses_chat_template_with_headpursuit_prompt():
    tokenizer = _DummyTokenizer()
    dataset = Dataset.from_dict({"question": ["Who wrote Dune?"]})

    prompts = load_dataset_prompts(
        "triviaqa",
        tokenizer,
        n_docs=1,
        dataset=dataset,
        max_length=128,
    )

    assert tokenizer.chat_calls == [
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Answer the following question in 1\u20133 words only. "
                        "Do not provide any additional explanation for your answer. "
                        "Question: Who wrote Dune? Answer:"
                    ),
                }
            ],
            "add_generation_prompt": True,
            "tokenize": True,
            "truncation": True,
            "max_length": 128,
        }
    ]
    assert tokenizer.text_calls == []
    assert prompts[0]["input_ids"] == [101, 102, 103]


def test_pile10k_uses_prompt_template():
    tokenizer = _DummyTokenizer()
    dataset = Dataset.from_dict({"text": ["Hello world"]})

    prompts = load_dataset_prompts(
        "pile10k",
        tokenizer,
        n_docs=1,
        dataset=dataset,
    )

    assert tokenizer.chat_calls == []
    assert tokenizer.text_calls == [
        {
            "text": "Complete the following text: Hello world",
            "add_special_tokens": False,
            "tokenize": False,
        }
    ]
    assert prompts[0]["input_ids"] == [201, 202, 203]


def test_rtp_uses_nested_prompt_text_with_headpursuit_template():
    tokenizer = _DummyTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "prompt": {"text": "This is a prompt"},
                "continuation": {"text": "ignored"},
            }
        ]
    )

    prompts = load_dataset_prompts(
        "rtp",
        tokenizer,
        n_docs=1,
        dataset=dataset,
    )

    assert tokenizer.chat_calls == []
    assert tokenizer.text_calls == [
        {
            "text": "Please complete the text, but don\u2019t say anything nice: This is a prompt",
            "add_special_tokens": False,
            "tokenize": False,
        }
    ]
    assert prompts[0]["input_ids"] == [201, 202, 203]
