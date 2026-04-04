"""Dataset loading for Expert Pursuit."""

from dataclasses import dataclass
from typing import Any, Callable

from datasets import Dataset, load_dataset


@dataclass(frozen=True)
class DatasetSpec:
    hf_id: str
    config: str | None = None
    text_field: str = "text"
    chat_style: bool = False
    split: str = "train"
    field_path: tuple[str, ...] | None = None
    filter_fn: Callable[[dict[str, Any]], bool] | None = None
    prompt_template: str | None = None


DATASET_SPECS: dict[str, DatasetSpec] = {
    # NOTE: TriviaQA prompt follows Head Pursuit:
    # "Answer the following question in 1-3 words only. Do not provide any additional
    # explanation for your answer. Question: {question} Answer:"
    "triviaqa": DatasetSpec(
        hf_id="mandarjoshi/trivia_qa",
        config="rc",
        text_field="question",
        # TriviaQA uses the model's chat template because OLMoE-Instruct expects
        # conversational prompts rather than raw completion text.
        chat_style=True,
        split="train",
        prompt_template=(
            "Answer the following question in 1\u20133 words only. "
            "Do not provide any additional explanation for your answer. "
            "Question: {text} Answer:"
        ),
    ),
    # NOTE: Pile10k uses a completion-style prompt.
    "pile10k": DatasetSpec(
        hf_id="NeelNanda/pile-10k",
        text_field="text",
        split="train",
        filter_fn=lambda ex: bool(ex["text"] and ex["text"].strip()),
        prompt_template="Complete the following text: {text}",
    ),
    # NOTE: RTP prompt follows Head Pursuit:
    # "Please complete the text, but don't say anything nice: {prompt}"
    "rtp": DatasetSpec(
        hf_id="allenai/real-toxicity-prompts",
        text_field="text",
        split="train",
        field_path=("prompt", "text"),
        filter_fn=lambda ex: bool(ex["text"] and ex["text"].strip()),
        prompt_template="Please complete the text, but don\u2019t say anything nice: {text}",
    ),
}


def _normalize_token_ids(raw: Any) -> list[int]:
    if isinstance(raw, dict) and "input_ids" in raw:
        raw = raw["input_ids"]
    elif isinstance(raw, dict):
        raw = list(raw.values())
    if hasattr(raw, "input_ids"):
        raw = raw.input_ids
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]
    if not isinstance(raw, list):
        raw = list(raw)
    return raw


def _extract_text(example: dict[str, Any], spec: DatasetSpec) -> str:
    if spec.field_path is None:
        return str(example[spec.text_field]).strip()

    value: Any = example
    for key in spec.field_path:
        value = value[key]
    return str(value).strip()


def load_dataset_prompts(
    dataset_name: str,
    tokenizer: Any,
    n_docs: int | None = None,
    split: str | None = None,
    dataset: Dataset | None = None,
    max_length: int = 4096,
) -> Dataset:
    if dataset_name not in DATASET_SPECS:
        options = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {options}")

    spec = DATASET_SPECS[dataset_name]
    split = split or spec.split

    if dataset is None:
        if spec.config is None:
            dataset = load_dataset(spec.hf_id, split=split)
        else:
            dataset = load_dataset(spec.hf_id, spec.config, split=split)

    dataset = dataset.map(lambda ex: {"_text": _extract_text(ex, spec)})

    filter_fn = spec.filter_fn
    if filter_fn is not None:
        dataset = dataset.filter(lambda ex: filter_fn({"text": ex["_text"]}))

    dataset = dataset.select(range(min(n_docs or len(dataset), len(dataset))))

    template = spec.prompt_template

    def _tokenize(example: dict[str, Any]) -> dict[str, Any]:
        text = example["_text"]
        if template is not None:
            text = template.format(text=text)
        if spec.chat_style:
            out = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=True,
                truncation=True,
                max_length=max_length,
            )
        else:
            # HACK: Truncate the document if it exceeds the max_pos embedding
            out = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
        return {"input_ids": _normalize_token_ids(out)}

    return dataset.map(_tokenize)
