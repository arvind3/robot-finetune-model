---
dataset_name: robotframework-expert-dataset
language:
- en
license: other
task_categories:
- text-generation
tags:
- robotframework
- automation
- synthetic
- documentation
created: 2026-02-23
---

# Dataset

This dataset is built from:

- Local Robot Framework documentation files in `sources/robotframework_docs/` (if present)
- Curated synthetic examples in `data/synthetic_examples.json`

## License and attribution

- Robot Framework docs remain under their original licenses. Do not redistribute doc-derived datasets unless the license allows it.
- Synthetic examples are authored for this project.

## Files

- `train.jsonl` and `eval.jsonl`: SFT records using `messages` format
- `build_meta.json`: counts and build configuration
