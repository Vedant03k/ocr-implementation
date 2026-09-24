import re

import torch

_SYSTEM_PROMPT = """You correct OCR misreads in numbered lines of text, using neighboring lines as context to resolve ambiguous words. Follow these rules exactly:
1. Fix only clear spelling/OCR errors (garbled, misspelled, or wrongly-split words).
2. Never replace a correctly-spelled word with a different word, even a synonym. Only touch words that are actually misspelled or garbled.
3. Never add, remove, merge, split, or reorder lines. Output exactly the same number of lines, in the same order.
4. Preserve capitalization and punctuation exactly, except when fixing the letters of a misspelled word.
5. If a line has no errors, output it completely unchanged.
6. Output ONLY the numbered lines in the same "N. text" format as the input. No explanation.

Example input:
1. Theme : to build to build a
2. seaules infrasture fir
3. our Industry to find out
4. Project Nave
5. "Shadan sentinel"
6. Subtotal: $842.00
7. problem
8. recvd by J.T. - ok to ship

Example output:
1. Theme : to build to build a
2. seamless infrastructure for
3. our Industry to find out
4. Project Name
5. "Shadow sentinel"
6. Subtotal: $842.00
7. problem
8. recvd by J.T. - ok to ship

Note in the example: line 5 keeps its quote marks, line 7 stays lowercase because it started lowercase, and line 8 is left untouched because "recvd" and "ok" are informal but not OCR errors — do not "improve" correctly-transcribed text."""

_LINE_PATTERN = re.compile(r"^\s*(\d+)\.\s?(.*)$")


class TextCleaner:
    """Small instruction-tuned LLM that corrects OCR typos across a document's lines.

    Cleans all lines in one call (numbered, so the model can use neighboring
    lines as context — a line cut mid-phrase by the line detector, like "fir"
    from "fir our Industry", is only disambiguated to "for" with that context)
    while still constraining it to preserve line count and order. Falls back
    to the original lines unchanged if the model doesn't return a matching
    line count, rather than risk silently dropping or merging content.
    """

    def __init__(self, model_dir: str, device: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None or (device.startswith("cuda") and not torch.cuda.is_available()):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir).to(self.device)

    def clean_lines(self, lines: list[str]) -> list[str]:
        if not any(line.strip() for line in lines):
            return list(lines)

        numbered_input = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": numbered_input},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=min(sum(len(line.split()) for line in lines) * 3 + 40, 1024),
            do_sample=False,
        )
        output = self.tokenizer.decode(
            generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        parsed = self._parse_numbered(output, expected=len(lines))
        return parsed if parsed is not None else list(lines)

    @staticmethod
    def _parse_numbered(output: str, expected: int) -> list[str] | None:
        result: dict[int, str] = {}
        for raw_line in output.strip().splitlines():
            match = _LINE_PATTERN.match(raw_line)
            if not match:
                continue
            index = int(match.group(1))
            result[index] = match.group(2)

        if len(result) != expected or set(result) != set(range(1, expected + 1)):
            return None
        return [result[i] for i in range(1, expected + 1)]
