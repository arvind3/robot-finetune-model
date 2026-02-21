from datetime import datetime

from tools.utils import base_model_short


def render_model_card(
    hf_repo: str,
    base_model: str,
    dataset_summary: str,
    limitations: str,
    usage: str,
) -> str:
    short = base_model_short(base_model)
    now = datetime.utcnow().strftime("%Y-%m-%d")
    return f"""---
license: apache-2.0
base_model: {base_model}
tags:
- robot-framework
- automation
- qlora
- lora
- unsloth
model_name: {hf_repo}
created: {now}
---

# {hf_repo}

## Overview

LoRA adapter fine-tuned from `{base_model}` for Robot Framework + Python automation tasks.

## Data Sources

{dataset_summary}

## Intended Use

{usage}

## Limitations

{limitations}

## How to Use

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "{base_model}"
adapter = "{hf_repo}"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)

messages = [
    {{"role": "system", "content": "You are an expert Robot Framework + Python automation engineer."}},
    {{"role": "user", "content": "Create a minimal Robot Framework test for login."}},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""
