"""Published task contracts and CLI scope; no model or dataset is loaded."""
import contextlib
import io
import unittest
from unittest.mock import patch
from benchmarks import generation
from benchmarks.contracts import TIMING_CONTRACT
from benchmarks.profiles import TASKS, task_protocol
from benchmarks.scoring import CODE_STOP_SEQUENCES


class PublishedTaskTests(unittest.TestCase):
    def test_prompt_and_scoring_contracts(self):
        expected = {
            "gsm8k": dict(kind="lm_eval", fewshot=4, chat=True, multiturn=True,
                          metric="exact_match,flexible-extract", max_gen=1024, expected_samples=1319),
            "humaneval_plus": dict(kind="evalplus", dataset="humaneval", fewshot=0,
                                   chat=False, multiturn=False, metric="pass_at_1_plus",
                                   max_gen=1024, expected_samples=164),
        }
        self.assertEqual(TASKS, expected)
        for platform, version in (("h20", "0.4.9.2"), ("rtx3090", "0.4.11")):
            for task, settings in expected.items():
                protocol = task_protocol(task, device_profile=platform)
                self.assertEqual({key: protocol[key] for key in settings}, settings)
                self.assertEqual(protocol["prompt_token_limit"], 1024)
                self.assertEqual(protocol["lm_eval_version"], version)
                self.assertEqual(protocol["stopping"], "EOS during generation; task stop strings applied after decoding")
        self.assertEqual(CODE_STOP_SEQUENCES, ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"])

    def test_timer_contract(self):
        self.assertEqual(TIMING_CONTRACT, dict(
            decode_numerator="timed_decode_tokens", decode_denominator="decode_time_seconds",
            request_numerator="raw_output_tokens", request_denominator="total_time_seconds",
            setup_in_throughput=False, scoring_in_runtime_throughput=False, ttft="runtime TTFT"))

    def test_cli_rejects_unpublished_tasks_before_setup(self):
        for task in ("ifeval", "math500", "mbpp_plus"):
            argv = ["generation", "--profile", "bonsai_1p7b_h20_d8_hybrid", "--task", task,
                    "--model-id", "test", "--checkpoint", "absent", "--tokenizer", "absent", "--output", "absent"]
            with patch("sys.argv", argv), patch.object(generation, "build_adapter") as build:
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error: generation.main()
                self.assertEqual(error.exception.code, 2)
                build.assert_not_called()

    def test_code_evaluation_rejects_other_datasets_before_import(self):
        with self.assertRaises(ValueError):
            generation.evaluate_code(None, "mbpp_plus", {"dataset": "mbpp"}, None)


if __name__ == "__main__":
    unittest.main()
