"""GSM8K and HumanEval+ scoring helpers; isolate generated-code execution."""
import os, sys, time, json, hashlib, subprocess, re, math
CODE_STOP_SEQUENCES = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
EVALPLUS_VERSION = "0.3.1"
HUMANEVAL_VERSION = "v0.1.10"
HUMANEVAL_HASH = "fe585eb4df8c88d844eeb463ea4d0302"
EVALPLUS_PARALLEL = 8
EVALPLUS_TIMEOUT = 600


def load_humaneval():
    """Load the fixed official dataset without executing any solutions."""
    from .task_dependencies import preflight_task_dependencies
    preflight_task_dependencies("humaneval_plus")
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
    from evalplus.data.humaneval import HUMANEVAL_PLUS_VERSION, HUMANEVAL_OVERRIDE_PATH
    if HUMANEVAL_PLUS_VERSION != HUMANEVAL_VERSION or HUMANEVAL_OVERRIDE_PATH:
        raise RuntimeError("HumanEval+ dataset version or override differs from the recipe")
    problems, dataset_hash = get_human_eval_plus(), get_human_eval_plus_hash()
    if dataset_hash != HUMANEVAL_HASH or len(problems) != 164:
        raise RuntimeError("official HumanEvalPlus-v0.1.10 dataset hash/cardinality mismatch")
    return problems, dataset_hash


_EVALPLUS_PASS1_RE = re.compile(r"pass@1\s*:\s*([0-9.]+)")


_EVALPLUS_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _evalplus_result_paths(samples_jsonl: str):
    return (
        samples_jsonl.replace(".jsonl", "_eval_results.json"),
        samples_jsonl.replace(".jsonl", ".eval_results.json"),
    )


def _evalplus_status_pass_at_1(data: dict):
    """Recompute EvalPlus v0.3.x pass@1 from its persisted status table.

    This mirrors evalplus.evaluate exactly for k=1.  The Plus score requires
    both the base and plus tests to pass, rather than looking at plus_status
    alone.  Averaging is per problem, so the implementation remains correct
    if a file contains more than one completion for a problem.
    """
    evaluations = data.get("eval")
    if not isinstance(evaluations, dict) or not evaluations:
        return None
    base_scores, plus_scores = [], []
    for rows in evaluations.values():
        if not isinstance(rows, list) or not rows:
            return None
        base_correct = 0
        plus_correct = 0
        for row in rows:
            if not isinstance(row, dict):
                return None
            base_status = row.get("base_status")
            plus_status = row.get("plus_status")
            if base_status is None or plus_status is None:
                return None
            base_pass = base_status == "pass"
            base_correct += int(base_pass)
            plus_correct += int(base_pass and plus_status == "pass")
        base_scores.append(base_correct / len(rows))
        plus_scores.append(plus_correct / len(rows))
    return (
        sum(base_scores) / len(base_scores),
        sum(plus_scores) / len(plus_scores),
    )


def _read_evalplus_pass_at_1_json(samples_jsonl: str):
    """Read pass@1 (base, plus) as 0–1 from evalplus JSON; None if unavailable.

    EvalPlus <=0.2 sometimes persisted a pass_at_k summary, whereas EvalPlus
    0.3.1 persists the per-problem ``eval`` status table and only prints the
    aggregate.  Support both schemas, but require exactly one valid candidate
    after the caller clears stale results.
    """
    valid_pairs = []
    for path in _evalplus_result_paths(samples_jsonl):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            pak = data.get("pass_at_k")
            pair = None
            if isinstance(pak, dict):
                base_d = pak.get("base")
                plus_d = pak.get("plus")
                if (isinstance(base_d, dict) and "pass@1" in base_d
                        and isinstance(plus_d, dict)
                        and "pass@1" in plus_d):
                    pair = (
                        float(base_d["pass@1"]),
                        float(plus_d["pass@1"]),
                    )
            if pair is None:
                pair = _evalplus_status_pass_at_1(data)
            if pair is None:
                continue
            valid_pairs.append(pair)
        except Exception:
            continue
    return valid_pairs[0] if len(valid_pairs) == 1 else None


def _parse_evalplus_pass_at_1_from_text(output: str):
    vals = []
    for line in output.split("\n"):
        line = _EVALPLUS_ANSI_RE.sub("", line)
        m = _EVALPLUS_PASS1_RE.search(line)
        if m:
            vals.append(float(m.group(1)))
    if len(vals) >= 2:
        return vals[0], vals[1]
    return None


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _created_evalplus_result_path(samples_jsonl: str):
    """Return the unique result created after cache clearing.

    The caller removes stale results before EvalPlus runs and checks that the
    new result contains the generated solutions. Filesystem timestamps are not used.
    """
    candidates = [
        path for path in _evalplus_result_paths(samples_jsonl)
        if os.path.isfile(path)
    ]
    if len(candidates) != 1:
        return None
    return candidates[0]


def _unlink_evalplus_result_caches(samples_jsonl: str) -> dict:
    """Remove stale result candidates and report which files were removed."""
    removed = []
    candidate_paths = list(_evalplus_result_paths(samples_jsonl))
    for path in candidate_paths:
        if os.path.isfile(path):
            removed.append({
                "path": path,
                "sha256": _sha256_file(path),
            })
            os.unlink(path)
    remaining = [path for path in candidate_paths if os.path.exists(path)]
    if remaining:
        raise RuntimeError(
            f"EvalPlus cache absence barrier failed: {remaining}")
    return {
        "method": "unlink_then_absence_barrier",
        "candidate_paths": candidate_paths,
        "removed": removed,
        "all_candidates_absent_after_clear": True,
    }


def _verify_evalplus_result_binding(
        samples_jsonl: str, result_json: str, problems: dict,
        expected_dataset_hash: str) -> dict:
    """Bind a persisted EvalPlus table exactly to generated samples.

    EvalPlus stores the full prompt+completion solution for every completion.
    Reconstruct those solutions from ``samples.jsonl`` and compare them with
    the persisted evaluator output.
    """
    expected = {}
    sample_rows = []
    with open(samples_jsonl, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            task_id = sample.get("task_id")
            if task_id not in problems:
                raise RuntimeError(
                    f"EvalPlus sample task mismatch at line {line_no}: "
                    f"{task_id!r}")
            if "solution" in sample:
                solution = sample["solution"]
            elif "completion" in sample:
                solution = problems[task_id]["prompt"] + sample["completion"]
            else:
                raise RuntimeError(
                    f"EvalPlus sample lacks solution at line {line_no}")
            expected.setdefault(task_id, []).append(solution)
            sample_rows.append([task_id, solution])

    with open(result_json, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("hash") != expected_dataset_hash:
        raise RuntimeError(
            f"EvalPlus dataset hash drift: {doc.get('hash')!r} != "
            f"{expected_dataset_hash!r}")
    evaluations = doc.get("eval")
    if not isinstance(evaluations, dict):
        raise RuntimeError("EvalPlus result lacks eval status table")
    if set(evaluations) != set(expected):
        raise RuntimeError("EvalPlus result task-id set does not match samples")

    result_rows = []
    for task_id, expected_solutions in expected.items():
        rows = evaluations.get(task_id)
        if not isinstance(rows, list):
            raise RuntimeError(
                f"EvalPlus result rows are invalid for {task_id}")
        actual_solutions = [row.get("solution") for row in rows]
        if actual_solutions != expected_solutions:
            raise RuntimeError(
                f"EvalPlus persisted solutions differ for {task_id}")
        for row in rows:
            if row.get("task_id") != task_id:
                raise RuntimeError(
                    f"EvalPlus row task-id drift for {task_id}")
            if row.get("base_status") is None \
                    or row.get("plus_status") is None:
                raise RuntimeError(
                    f"EvalPlus row status missing for {task_id}")
            result_rows.append([task_id, row.get("solution")])

    sample_payload = json.dumps(
        sample_rows, ensure_ascii=False, separators=(",", ":")).encode()
    result_payload = json.dumps(
        result_rows, ensure_ascii=False, separators=(",", ":")).encode()
    sample_binding_sha = hashlib.sha256(sample_payload).hexdigest()
    result_binding_sha = hashlib.sha256(result_payload).hexdigest()
    if sample_binding_sha != result_binding_sha:
        raise RuntimeError("EvalPlus solution binding SHA mismatch")
    return {
        "method": "exact_task_id_and_full_solution_sequence",
        "dataset_hash": doc.get("hash"),
        "expected_dataset_hash": expected_dataset_hash,
        "dataset_hash_match": True,
        "problem_count": len(expected),
        "completion_count": len(sample_rows),
        "task_id_sets_match": True,
        "solution_sequences_exact": True,
        "solution_binding_sha256": sample_binding_sha,
    }


def _eval_evalplus(dataset, samples_file):
    """Run evalplus.evaluate; return (pass@1 base, pass@1 plus) as 0–1 floats."""
    cmd = [sys.executable, "-m", "evalplus.evaluate",
           "--dataset", dataset, "--samples", samples_file,
           "--parallel", str(EVALPLUS_PARALLEL)]
    transcript = samples_file.replace(".jsonl", "_eval_stdout.txt")
    # Redirect to disk so a timeout still leaves the evaluator's diagnostics.
    # Terminate its process group as EvalPlus creates child workers.
    with open(transcript, "x", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        try:
            proc.wait(timeout=EVALPLUS_TIMEOUT)
        except BaseException:
            import signal
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise
    with open(transcript, encoding="utf-8") as f:
        out = f.read()
    if proc.returncode != 0:
        raise RuntimeError(
            f"EvalPlus failed for {dataset} with rc={proc.returncode}; "
            f"see {transcript}")
    reported = _parse_evalplus_pass_at_1_from_text(out)
    if reported is None:
        raise RuntimeError(
            f"EvalPlus did not report both base and plus pass@1 for "
            f"{dataset}; see {transcript}")
    return reported


def _truncate_completion(text, stop_seqs):
    best = len(text)
    for s in stop_seqs:
        idx = text.find(s)
        if idx != -1 and idx < best:
            best = idx
    return text[:best]


def _extract_acc(task_res, task_name, preferred_metric=None):
    if preferred_metric is None or preferred_metric not in task_res:
        raise ValueError(f"{task_name}: required metric {preferred_metric!r} is missing")
    value = task_res[preferred_metric]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{task_name}: required accuracy must be a finite fraction")
    return float(value) * 100.0

