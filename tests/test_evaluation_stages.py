"""CPU mocks only: no model, dataset download, or generated Python execution."""
import copy
import contextlib
import io
import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks import generation, scoring, score_humaneval
from benchmarks.contracts import TIMING_CONTRACT, _digest_manifest
from benchmarks.profiles import PROFILES, task_protocol


class FakeWrap:
    def __init__(self):
        self._acc_cnt = 0
        self._out_tok = 0
        self._request_receipts = []

    def gen_code(self, prompt, max_toks):
        assert max_toks == 1024
        self._acc_cnt += 1
        self._out_tok += 2
        self._request_receipts.append(dict(raw_output_token_ids=[10, 20], raw_output_token_count=2,
            timed_decode_tokens=1, decode_time_seconds=0.02, total_time_seconds=0.03))
        return "    return 1\nprint('must never execute')"

    def get_stats(self):
        return dict(timed_decode_tokens=self._acc_cnt, decode_time_seconds=self._acc_cnt * 0.02,
                    total_time_seconds=self._acc_cnt * 0.03, sequence_record_count=self._acc_cnt,
                    decode_speed=50.0 if self._acc_cnt else 0.0)


class EvaluationStages(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "generated"
        self.source.mkdir()
        self.problems = {f"HumanEval/{i}": {"prompt": f"def f{i}():\n"} for i in range(164)}
        files = {"fixture": {"size_bytes": 1, "sha256": "0" * 64}}
        identity = {r: dict(files=copy.deepcopy(files), sha256=_digest_manifest(files))
                    for r in ("model", "tokenizer", "implementation")}
        name = "bonsai_1p7b_h20_d8_hybrid"
        self.base = dict(schema="bitmtp_generation_v2", task="humaneval_plus", family="bonsai",
            model_id="test-model", profile=name, profile_contract=PROFILES[name], checkpoint_role="mtp",
            artifact_identity=identity, artifact_identity_seconds=0.1,
            task_protocol=task_protocol("humaneval_plus"), setup_and_warmup_seconds=0.2,
            timing_contract=dict(TIMING_CONTRACT), hardware=dict(gpu="NVIDIA H20", capability=[9, 0],
                torch="2.10.0+cu128", python="3.10.18", lm_eval="0.4.9.2"))
        patcher = patch.object(scoring, "load_humaneval", return_value=(self.problems, scoring.HUMANEVAL_HASH))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.wrap = FakeWrap()

    def generate(self, only=True):
        started = time.perf_counter()
        return generation.evaluate_code(self.wrap, "humaneval_plus", self.base["task_protocol"], self.source,
            generate_only=only, started=started,
            on_generated=lambda ids, receipt: generation.persist_generation(
                self.wrap, self.source, self.base, ids, receipt, started))

    def mock_evaluator(self, dataset, samples):
        samples = Path(samples)
        evaluations = {}
        for i, row in enumerate(map(json.loads, samples.read_text().splitlines())):
            task = row["task_id"]
            evaluations[task] = [dict(task_id=task, solution=self.problems[task]["prompt"] + row["completion"],
                base_status="fail" if i == 0 else "pass", plus_status="fail" if i == 1 else "pass")]
        samples.with_name("samples_eval_results.json").write_text(json.dumps(
            dict(hash=scoring.HUMANEVAL_HASH, eval=evaluations)))
        samples.with_name("samples_eval_stdout.txt").write_text("fresh mock evaluator\n")
        return 163 / 164, 162 / 164

    def test_generate_only_never_scores_or_claims_pass(self):
        with patch.object(scoring, "_eval_evalplus") as evaluator:
            record = self.generate()
        evaluator.assert_not_called()
        self.assertEqual(record["status"], "GENERATED_UNSCORED")
        self.assertEqual(record["observed_samples"], 164)
        self.assertNotIn("score_percent", record)
        self.assertFalse((self.source / "result.json").exists())
        self.assertNotIn("print", (self.source / "samples.jsonl").read_text())
        self.assertEqual(record["stats"]["raw_output_tokens"], 328)

    def test_cli_generation_only_does_not_require_code_execution_acknowledgement(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "bonsai_gpu_config.json").write_text("{}")
        argv = ["generation", "--profile", self.base["profile"], "--task", "humaneval_plus",
                "--model-id", "test-model", "--checkpoint", str(checkpoint), "--tokenizer", "unused",
                "--output", str(self.root / "new"), "--generate-only", "--check-only"]
        with patch("sys.argv", argv), patch("benchmarks.task_dependencies.preflight_task_dependencies"), \
                patch("importlib.metadata.version", return_value="0.4.9.2"), \
                patch.object(generation, "validate_bonsai_config"), patch.object(generation, "artifact_files"), \
                patch.object(generation, "build_adapter") as build, contextlib.redirect_stdout(io.StringIO()) as out:
            generation.main()
        build.assert_not_called()
        self.assertFalse(json.loads(out.getvalue())["gpu_executed"])

    def test_default_scoring_failure_preserves_generation_facts(self):
        def fail(*args):
            self.assertTrue((self.source / "generation.json").is_file())
            self.assertTrue((self.source / "request_receipts.json").is_file())
            raise RuntimeError("mock worker fork failure")
        with patch.object(scoring, "_eval_evalplus", side_effect=fail), self.assertRaisesRegex(RuntimeError, "fork failure"):
            self.generate(only=False)
        self.assertFalse((self.source / "result.json").exists())
        self.assertEqual(json.loads((self.source / "scoring_failure.json").read_text())["status"], "SCORING_FAILED")
        self.assertEqual(score_humaneval.validate_generation(self.source, self.problems, scoring.HUMANEVAL_HASH)["stats"]["raw_output_tokens"], 328)

    def test_cli_rejects_nonstandard_humaneval_cap_before_setup(self):
        for cap in (0, 1, 512, 1025, 2048):
            argv = ["generation", "--profile", self.base["profile"], "--task", "humaneval_plus",
                    "--model-id", "test-model", "--checkpoint", "unused", "--tokenizer", "unused",
                    "--output", str(self.root / "new"), "--generate-only", "--max-gen", str(cap)]
            with self.subTest(cap=cap), patch("sys.argv", argv), \
                    patch.object(generation, "build_adapter") as build, \
                    contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as raised:
                generation.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("requires --max-gen 1024", err.getvalue())
            build.assert_not_called()

    def test_single_statistic_mutations_rejected_before_scoring(self):
        self.generate()
        path = self.source / "generation.json"
        original = json.loads(path.read_text())
        mutations = [("sequence_record_count", 163), ("sequence_record_count", 164.0),
                     ("decode_speed", 51), ("decode_speed", float("inf")),
                     ("decode_speed", float("nan")), ("raw_request_tok_s", 1),
                     ("raw_request_tok_s", float("inf")), ("raw_request_tok_s", float("nan"))]
        for key, value in mutations:
            changed = copy.deepcopy(original)
            changed["stats"][key] = value
            path.write_text(json.dumps(changed))
            with self.subTest(key=key, value=value), patch.object(scoring, "_eval_evalplus") as evaluator, \
                    self.assertRaisesRegex(ValueError, key):
                score_humaneval.score_record(self.source, self.root / "score")
            evaluator.assert_not_called()

    def test_two_stage_score_counts_binding_and_generation_hardware(self):
        before = self.generate()
        input_files = {p.name: p.read_bytes() for p in self.source.iterdir()}
        with patch.object(scoring, "_eval_evalplus", side_effect=self.mock_evaluator), patch.object(score_humaneval, "version", return_value="2.2.6"):
            result = score_humaneval.score_record(self.source, self.root / "score")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["raw_metrics"]["base_correct"], 163)
        self.assertEqual(result["raw_metrics"]["plus_correct"], 162)
        self.assertEqual(result["stats"], before["stats"])
        self.assertEqual(result["hardware"], before["hardware"])
        self.assertIsNone(result["wall_time_seconds"])
        self.assertTrue(result["evalplus_receipt"]["exact_solution_binding"]["solution_sequences_exact"])
        self.assertEqual(input_files, {p.name: p.read_bytes() for p in self.source.iterdir()})

    def test_mutated_or_old_samples_rejected_before_execution(self):
        self.generate()
        path = self.source / "samples.jsonl"
        path.write_text(path.read_text().replace("return 1", "return 2", 1))
        with patch.object(scoring, "_eval_evalplus") as evaluator, self.assertRaisesRegex(ValueError, "SHA256"):
            score_humaneval.score_record(self.source, self.root / "score")
        evaluator.assert_not_called()

    def test_receipt_mutation_rejected(self):
        self.generate()
        path = self.source / "request_receipts.json"
        path.write_text("[]")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            score_humaneval.validate_generation(self.source, self.problems, scoring.HUMANEVAL_HASH)

    def test_protocol_identity_and_false_pass_rejected(self):
        self.generate()
        path = self.source / "generation.json"
        original = json.loads(path.read_text())
        mutations = [("status", "PASS"), ("task_protocol", dict(original["task_protocol"], max_gen=2048)),
                     ("hardware", dict(original["hardware"], gpu="RTX 3090")),
                     ("artifact_identity", {}), ("score_percent", 100),
                     ("generation_receipt", dict(original["generation_receipt"], dataset_hash="old"))]
        for key, value in mutations:
            path.write_text(json.dumps(dict(original, **{key: value})))
            with self.subTest(key=key), self.assertRaises(ValueError):
                score_humaneval.validate_generation(self.source, self.problems, scoring.HUMANEVAL_HASH)

    def test_duplicate_ids_rejected_even_with_updated_sample_hash(self):
        self.generate()
        samples = self.source / "samples.jsonl"
        rows = samples.read_text().splitlines()
        rows[-1] = rows[0]
        samples.write_text("\n".join(rows) + "\n")
        path = self.source / "generation.json"
        record = json.loads(path.read_text())
        record["generation_receipt"]["sample_sha256"] = scoring._sha256_file(samples)
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "cardinality"):
            score_humaneval.validate_generation(self.source, self.problems, scoring.HUMANEVAL_HASH)

    def test_stale_output_is_not_reused(self):
        self.generate()
        output = self.root / "score"
        output.mkdir()
        with patch.object(scoring, "_eval_evalplus") as evaluator, self.assertRaises(FileExistsError):
            score_humaneval.score_record(self.source, output)
        evaluator.assert_not_called()

    def test_wrong_solution_binding_cannot_create_pass(self):
        self.generate()
        def wrong(dataset, samples):
            result = self.mock_evaluator(dataset, samples)
            path = Path(samples).with_name("samples_eval_results.json")
            path.write_text(path.read_text().replace("return 1", "return 9", 1))
            return result
        with patch.object(scoring, "_eval_evalplus", side_effect=wrong), self.assertRaisesRegex(RuntimeError, "solutions differ"):
            score_humaneval.score_record(self.source, self.root / "score")
        self.assertFalse((self.root / "score/result.json").exists())

    def test_evaluator_uses_eight_workers_and_preserves_timeout_transcript(self):
        path = self.root / "timeout.jsonl"
        proc = SimpleNamespace(pid=123, wait=unittest.mock.Mock(side_effect=[subprocess.TimeoutExpired("evalplus", 600), 0]))
        with patch.object(scoring.subprocess, "Popen", return_value=proc) as popen, patch.object(scoring.os, "killpg") as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                scoring._eval_evalplus("humaneval", str(path))
        self.assertEqual(popen.call_args.args[0][-2:], ["--parallel", "8"])
        proc.wait.assert_any_call(timeout=600)
        kill.assert_called_once()
        self.assertTrue(path.with_name("timeout_eval_stdout.txt").is_file())


if __name__ == "__main__":
    unittest.main()
