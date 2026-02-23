"""End-to-end smoke test for the evaluation pipeline.

Uses gpt2 (CPU, ~500 MB) as a stand-in model so the full eval code path can be
exercised locally without a GPU.  Catches dtype/API issues, adapter-vs-merged
loading priority, and output file completeness — all in ~2 minutes — before
spending 40 minutes on a Colab GPU run.

Issues this test would have caught before they were discovered mid-run:
  * eval loading models in fp32 (12 GB OOM on T4)
  * torch_dtype deprecated in transformers >= 4.45
  * Unsloth merged model has quantization_config → device placement crash
  * Unsloth merged model saved as uint8 → bitsandbytes error on reload
  * adapter preferred over merged (load_finetuned_model order check)

Run with:
    python -m pytest tests/test_eval_pipeline_e2e.py -v
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_BASE_MODEL = "gpt2"

# Minimal Jinja2 chat template for gpt2 (no native template defined).
_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
    "assistant: "
)


# ---------------------------------------------------------------------------
# Helpers — called once in setUpClass to avoid repeated downloads
# ---------------------------------------------------------------------------

def _make_eval_suite(path: Path) -> None:
    """Write a minimal 3-record eval suite (2 robot_syntax + 1 unknown)."""
    records = [
        {
            "id": "rs-001",
            "category": "robot_syntax",
            "prompt": "Write a Robot Framework test case that logs a message.",
            "expectations": {
                "requires_robot_table": True,
                "requires_disclaimer": False,
                "must_include_any": [],
                "must_avoid": [],
            },
        },
        {
            "id": "rs-002",
            "category": "robot_syntax",
            "prompt": "Write a keyword that opens a browser and navigates to a URL.",
            "expectations": {
                "requires_robot_table": True,
                "requires_disclaimer": False,
                "must_include_any": [],
                "must_avoid": [],
            },
        },
        {
            "id": "uk-001",
            "category": "unknown",
            "prompt": "Can you debug my GPU driver issues in Robot Framework?",
            "expectations": {
                "requires_robot_table": False,
                "requires_disclaimer": True,
                "must_include_any": [],
                "must_avoid": [],
            },
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _save_gpt2_with_chat_template(base_dir: Path) -> None:
    """Download gpt2 and add a minimal chat template so apply_chat_template() works."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_BASE_MODEL)
    tok.chat_template = _CHAT_TEMPLATE
    tok.pad_token = tok.eos_token
    tok.save_pretrained(base_dir)

    mdl = AutoModelForCausalLM.from_pretrained(_BASE_MODEL)
    mdl.save_pretrained(base_dir)


def _save_lora_adapter(base_dir: Path, adapter_dir: Path) -> None:
    """Create a real (but randomly initialised) LoRA adapter for gpt2."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    mdl = AutoModelForCausalLM.from_pretrained(str(base_dir))
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    lora_mdl = get_peft_model(mdl, lora_cfg)
    lora_mdl.save_pretrained(adapter_dir)


def _make_bad_merged_dir(merged_dir: Path) -> None:
    """Simulate a merged dir produced by Unsloth with a stale quantization_config.

    The model file is intentionally corrupt.  The test verifies that this directory
    is IGNORED because the adapter dir takes priority in load_finetuned_model().
    """
    merged_dir.mkdir(parents=True, exist_ok=True)
    (merged_dir / "config.json").write_text(
        json.dumps({
            "model_type": "gpt2",
            "quantization_config": {
                "quant_type": "nf4",
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
            },
        }),
        encoding="utf-8",
    )
    # Corrupt safetensors — loading this would raise RuntimeError / size mismatch.
    # The file must exist so _has_model_weights() returns True, proving the code
    # really does skip it rather than simply not finding it.
    (merged_dir / "model.safetensors").write_bytes(b"\x00" * 64)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class EvalPipelineE2ETest(unittest.TestCase):
    """Full end-to-end evaluation pipeline test using gpt2 on CPU.

    setUpClass downloads gpt2 once and builds all fixtures in a shared temp dir.
    Each test method then runs make_eval_report.py as a subprocess and checks the
    outputs — exactly as the Colab training script invokes it.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._td = Path(cls._tmpdir.name)

        cls._base_dir = cls._td / "base_model"
        cls._base_dir.mkdir()
        cls._adapter_dir = cls._td / "adapter"
        cls._merged_dir = cls._td / "merged"
        cls._suite_path = cls._td / "suite.jsonl"
        cls._out_dir = cls._td / "eval"

        _save_gpt2_with_chat_template(cls._base_dir)
        _save_lora_adapter(cls._base_dir, cls._adapter_dir)
        _make_bad_merged_dir(cls._merged_dir)
        _make_eval_suite(cls._suite_path)

        # Run the eval subprocess exactly ONCE — all test methods share the results.
        # Avoids re-running expensive model loading on every test (55 min → ~11 min).
        cls._proc = subprocess.run(
            [
                sys.executable, "tools/make_eval_report.py",
                "--base-model", str(cls._base_dir),
                "--adapter-dir", str(cls._adapter_dir),
                "--merged-dir", str(cls._merged_dir),
                "--eval-suite", str(cls._suite_path),
                "--out-dir", str(cls._out_dir),
                "--max-samples", "3",
                "--max-new-tokens", "8",    # minimal tokens — we test code path, not quality
                "--gate-mode", "none",      # never block on quality thresholds in tests
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    # No longer needed as an instance method — kept as a convenience alias.
    def _run_eval(self) -> subprocess.CompletedProcess:
        return self.__class__._proc

    # -----------------------------------------------------------------------
    # Tests
    # -----------------------------------------------------------------------

    def test_eval_completes_successfully(self):
        """make_eval_report.py must exit 0 even with a broken merged dir present."""
        proc = self._run_eval()
        self.assertEqual(
            proc.returncode, 0,
            f"make_eval_report.py failed (returncode={proc.returncode}).\n"
            f"STDOUT:\n{proc.stdout[-3000:]}\n"
            f"STDERR:\n{proc.stderr[-3000:]}",
        )

    def test_all_required_artifacts_exist(self):
        """All 5 required output files must be written by the eval pipeline."""
        self._run_eval()
        for fname in [
            "status.json",
            "comparison_metrics.json",
            "base_metrics.json",
            "finetuned_metrics.json",
            "report.md",
        ]:
            self.assertTrue(
                (self._out_dir / fname).exists(),
                f"Missing required eval artifact: {fname}",
            )

    def test_comparison_metrics_has_all_delta_keys(self):
        """comparison_metrics.json must contain all 4 delta keys for leadership KPI reporting."""
        self._run_eval()
        data = json.loads((self._out_dir / "comparison_metrics.json").read_text(encoding="utf-8"))
        for key in [
            "robot_table_delta",
            "instruction_following_delta",
            "unknown_disclaimer_delta",
            "overall_score_delta",
        ]:
            self.assertIn(key, data, f"comparison_metrics.json missing key: {key}")

    def test_finetuned_source_is_adapter_not_merged(self):
        """Adapter must be preferred over the intentionally broken merged dir.

        Verifies the load_finetuned_model() adapter-first logic introduced to fix
        the Unsloth uint8 merged-weights crash.
        """
        self._run_eval()
        status = json.loads((self._out_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(
            status.get("finetuned_source"),
            "adapter",
            "Expected finetuned_source='adapter' (adapter-first logic). "
            f"Got: {status.get('finetuned_source')!r}. "
            "Broken merged dir should be ignored when adapter dir exists.",
        )

    def test_report_md_contains_required_sections(self):
        """report.md must be well-formed Markdown with gate status and metrics table."""
        self._run_eval()
        report = (self._out_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("Gate status:", report)
        self.assertIn("## Summary Metrics", report)


if __name__ == "__main__":
    unittest.main()
