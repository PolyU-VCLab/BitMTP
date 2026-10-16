"""Score saved HumanEval+ generations in a separate, isolated CPU environment."""
import argparse
import json
import math
import platform
import shutil
import time
from importlib.metadata import version
from pathlib import Path

from . import scoring
from .contracts import TIMING_CONTRACT, validate_artifact_identity, validate_hardware_record
from .profiles import PROFILES, task_protocol, validate_profile


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def validate_generation(source, problems, dataset_hash):
    """Validate saved facts; never load model files or reconstruct runtime timers."""
    record = json.loads((source / "generation.json").read_text())
    if record.get("schema") != "bitmtp_generation_v2" or record.get("status") != "GENERATED_UNSCORED":
        raise ValueError("expected a GENERATED_UNSCORED generation.json")
    if any(key in record for key in ("score_percent", "raw_metrics", "evalplus_receipt")):
        raise ValueError("unscored generation contains scoring fields")
    profile = validate_profile(record["profile_contract"])
    builtin = PROFILES.get(record.get("profile"))
    if builtin is None or any(profile[k] != builtin[k] for k in (
            "family", "model_size", "checkpoint_role", "reference_model_id", "mode",
            "block_size", "device_profile", "verification", "lm_eval_version",
            "replay_widths", "segment_ladder")):
        raise ValueError("profile identity mismatch")
    if (record.get("task") != "humaneval_plus" or record.get("family") != profile["family"]
            or record.get("checkpoint_role") != profile["checkpoint_role"]
            or not isinstance(record.get("model_id"), str) or not record["model_id"]):
        raise ValueError("generation identity mismatch")
    if record["task_protocol"] != task_protocol("humaneval_plus", device_profile=profile["device_profile"]):
        raise ValueError("generation protocol differs from the published recipe")
    validate_artifact_identity(record.get("artifact_identity"))
    validate_hardware_record(profile, record.get("hardware"))
    if record.get("timing_contract") != TIMING_CONTRACT:
        raise ValueError("generation timing contract mismatch")
    receipt = record["generation_receipt"]
    if (receipt.get("dataset_hash") != dataset_hash or dataset_hash != scoring.HUMANEVAL_HASH
            or receipt.get("dataset_version") != scoring.HUMANEVAL_VERSION
            or receipt.get("evalplus_version") != scoring.EVALPLUS_VERSION):
        raise ValueError("generation dataset/evaluator identity mismatch")
    for name, key in (("samples.jsonl", "sample_sha256"),
                      ("request_receipts.json", "request_receipts_sha256")):
        if scoring._sha256_file(source / name) != receipt.get(key):
            raise ValueError(f"{name} SHA256 mismatch")
    samples = [json.loads(line) for line in (source / "samples.jsonl").read_text().splitlines()]
    ids = [row.get("task_id") for row in samples]
    if (len(problems) != 164 or len(ids) != 164 or len(set(ids)) != 164
            or set(ids) != set(problems) or record.get("observed_samples") != 164):
        raise ValueError("generation sample cardinality/task IDs mismatch")
    if any(set(row) != {"task_id", "completion"} or not isinstance(row["completion"], str) for row in samples):
        raise ValueError("expected one unsanitized completion per HumanEval task")
    receipts = json.loads((source / "request_receipts.json").read_text())
    if len(receipts) != 164 or [r.get("task_id") for r in receipts] != ids:
        raise ValueError("generation request receipt cardinality/order mismatch")
    stats = record["stats"]
    if type(stats.get("sequence_record_count")) is not int or stats["sequence_record_count"] != 164:
        raise ValueError("generation sequence_record_count must be 164")
    for key in ("raw_output_tokens", "timed_decode_tokens"):
        receipt_key = "raw_output_token_count" if key == "raw_output_tokens" else key
        if (type(stats[key]) is not int or stats[key] < 0
                or any(type(r[receipt_key]) is not int or r[receipt_key] < 0 for r in receipts)
                or sum(r[receipt_key] for r in receipts) != stats[key]):
            raise ValueError(f"generation {key} receipt mismatch")
    if any(len(r["raw_output_token_ids"]) != r["raw_output_token_count"] for r in receipts):
        raise ValueError("generation raw token IDs/count mismatch")
    for key in ("decode_time_seconds", "total_time_seconds"):
        if (not math.isfinite(stats[key]) or stats[key] <= 0
                or any(not math.isfinite(r[key]) or r[key] < 0 for r in receipts)
                or not math.isclose(sum(r[key] for r in receipts), stats[key], rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError(f"generation {key} receipt mismatch")
    for key, numerator, denominator in (
            ("decode_speed", "timed_decode_tokens", "decode_time_seconds"),
            ("raw_request_tok_s", "raw_output_tokens", "total_time_seconds")):
        value = stats.get(key)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not math.isclose(value, stats[numerator] / stats[denominator], rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError(f"generation {key} does not match its runtime counters")
    for key in ("generation_wall_time_seconds", "setup_and_warmup_seconds", "artifact_identity_seconds"):
        if not math.isfinite(record[key]) or record[key] < 0:
            raise ValueError(f"invalid generation time: {key}")
    return record


def score_record(source, output, *, standalone=True, pipeline_started=None):
    """Input is read-only. An in-process caller may use its generation output."""
    source, output = Path(source), Path(output)
    problems, dataset_hash = scoring.load_humaneval()
    generation = validate_generation(source, problems, dataset_hash)
    if standalone:
        if source.resolve() == output.resolve() or source.resolve() in output.resolve().parents:
            raise ValueError("scoring output must be separate from the generation input")
        output.mkdir(parents=True, exist_ok=False)
        for name in ("generation.json", "samples.jsonl", "request_receipts.json"):
            shutil.copyfile(source / name, output / name)
    sample_path = output / "samples.jsonl"
    # Check the copies as well, before executing any generated code.
    validate_generation(output, problems, dataset_hash)
    started = time.perf_counter()
    try:
        candidates = scoring._evalplus_result_paths(str(sample_path))
        if any(Path(p).exists() for p in candidates):
            raise RuntimeError("scoring output already contains EvalPlus results")
        reported = scoring._eval_evalplus("humaneval", str(sample_path))
        result_path = scoring._created_evalplus_result_path(str(sample_path))
        if result_path is None:
            raise RuntimeError("EvalPlus did not persist exactly one fresh result")
        if "Load from previous results" in (output / "samples_eval_stdout.txt").read_text():
            raise RuntimeError("EvalPlus reused a previous result")
        binding = scoring._verify_evalplus_result_binding(str(sample_path), result_path, problems, dataset_hash)
        if binding["problem_count"] != 164 or binding["completion_count"] != 164:
            raise RuntimeError("EvalPlus result/sample binding count mismatch")
        doc = json.loads(Path(result_path).read_text())
        rows = [rows[0] for rows in doc["eval"].values()]
        if any(r[k] not in {"pass", "fail", "timeout"} for r in rows for k in ("base_status", "plus_status")):
            raise RuntimeError("EvalPlus persisted an unknown status")
        base_correct = sum(r["base_status"] == "pass" for r in rows)
        plus_correct = sum(r["base_status"] == r["plus_status"] == "pass" for r in rows)
        persisted = (base_correct / 164, plus_correct / 164)
        if any(abs(a - b) > 0.00051 for a, b in zip(reported, persisted)):
            raise RuntimeError("EvalPlus printed and persisted scores disagree")
        validate_generation(output, problems, dataset_hash)
        if standalone:
            validate_generation(source, problems, dataset_hash)
            if scoring._sha256_file(source / "generation.json") != scoring._sha256_file(output / "generation.json"):
                raise RuntimeError("generation identity changed during scoring")
        elapsed = time.perf_counter() - started
        receipt = dict(dataset_hash=dataset_hash, dataset_version=scoring.HUMANEVAL_VERSION,
            evaluator_version=scoring.EVALPLUS_VERSION,
            sample_sha256=generation["generation_receipt"]["sample_sha256"],
            result_sha256=scoring._sha256_file(result_path), exact_solution_binding=binding,
            base_correct=base_correct, plus_correct=plus_correct, total=164,
            parallel=scoring.EVALPLUS_PARALLEL, total_timeout_seconds=scoring.EVALPLUS_TIMEOUT,
            per_problem_limits="EvalPlus 0.3.1 defaults",
            environment=dict(python=platform.python_version(), numpy=version("numpy"), platform=platform.platform()),
            generation_record_sha256=scoring._sha256_file(output / "generation.json"))
        result = dict(generation, status="PASS", score_percent=persisted[1] * 100,
            raw_metrics=dict(pass_at_1_base=persisted[0], pass_at_1_plus=persisted[1],
                             base_correct=base_correct, plus_correct=plus_correct),
            evalplus_receipt=receipt, scoring_wall_time_seconds=elapsed,
            wall_time_seconds=None if pipeline_started is None else time.perf_counter() - pipeline_started,
            execution_stages="separate_generation_and_scoring" if standalone else "single_process_generation_and_scoring")
        write_json(output / "result.json", result)
        return result
    except Exception as error:
        write_json(output / "scoring_failure.json", dict(status="SCORING_FAILED", error_type=type(error).__name__,
            error=str(error), scoring_wall_time_seconds=time.perf_counter() - started,
            generation_status="GENERATED_UNSCORED"))
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", required=True, help="Read-only directory containing generation.json and samples")
    ap.add_argument("--output", required=True, help="New directory for scoring outputs")
    ap.add_argument("--allow-code-execution", action="store_true", help="Acknowledgement, not a security sandbox")
    args = ap.parse_args(argv)
    if not args.allow_code_execution:
        ap.error("use --allow-code-execution only inside an isolated scoring environment")
    result = score_record(args.generation, args.output)
    print(json.dumps(dict(status=result["status"], score_percent=result["score_percent"],
                         raw_metrics=result["raw_metrics"]), indent=2))


if __name__ == "__main__":
    main()
