"""Artifact identity and timer definitions for benchmark results."""
import hashlib
import json
import re
from pathlib import Path

TIMING_CONTRACT = dict(
    decode_numerator="timed_decode_tokens", decode_denominator="decode_time_seconds",
    request_numerator="raw_output_tokens", request_denominator="total_time_seconds",
    setup_in_throughput=False, scoring_in_runtime_throughput=False, ttft="runtime TTFT")

TOKENIZER_FILES = (
    "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json", "config.json",
    "chat_template.jinja", "chat_template.json")

def artifact_files(checkpoint, tokenizer):
    """Check the local files used by the packed model and Qwen tokenizer."""
    checkpoint, tokenizer = Path(checkpoint), Path(tokenizer)
    model = {name: checkpoint/name for name in ("bonsai_gpu_config.json", "model_state_t2g128.pt")}
    for path in model.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required nonempty packed model file is missing: {path}")
    if not tokenizer.is_dir():
        raise ValueError("--tokenizer must be a local directory with the matching Qwen tokenizer")
    required_tokenizer = ["tokenizer_config.json"]
    if (tokenizer/"tokenizer.json").is_file():
        required_tokenizer.append("tokenizer.json")
    elif all((tokenizer/name).is_file() for name in ("vocab.json", "merges.txt")):
        required_tokenizer.extend(("vocab.json", "merges.txt"))
    else:
        raise ValueError("tokenizer.json or both vocab.json and merges.txt are required")
    for name in required_tokenizer:
        path = tokenizer/name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required nonempty tokenizer file is missing: {name}")
    token_files = {name: tokenizer/name for name in TOKENIZER_FILES if (tokenizer/name).is_file()}
    templates = tokenizer/"chat_templates"
    if templates.is_dir():
        token_files.update({str(p.relative_to(tokenizer)): p for p in templates.rglob("*") if p.is_file()})
    # Custom tokenizer implementations are executable inputs as well.
    token_files.update({str(p.relative_to(tokenizer)): p for p in tokenizer.glob("*.py") if p.is_file()})
    return {"model": model, "tokenizer": token_files}

def _digest_manifest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def implementation_files():
    """Select benchmark/runtime sources and the native library used by the loader."""
    import importlib.util
    gpu_root = Path(importlib.util.find_spec("bonsai_sft.gpu").origin).parent
    benchmark_root = Path(__file__).resolve().parent
    return implementation_files_from_roots(gpu_root, benchmark_root)

def implementation_files_from_roots(gpu_root, benchmark_root):
    files = {}
    for root, prefix, suffixes in (
        (Path(gpu_root), "bonsai_sft/gpu", {".py", ".cu", ".cuh", ".h", ".sh"}),
        (Path(benchmark_root), "benchmarks", {".py", ".yaml"}),
    ):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts:
                files[f"{prefix}/{path.relative_to(root)}"] = path
    library = Path(gpu_root)/"t2g128_kernels/libbonsai_t2.so"
    if not library.is_file() or library.stat().st_size == 0:
        raise ValueError("build t2g128_kernels/libbonsai_t2.so before running a benchmark")
    files["bonsai_sft/gpu/t2g128_kernels/libbonsai_t2.so"] = library
    return {"implementation": files}

def fingerprint_artifacts(files):
    """Read each artifact once, before model setup and benchmark timing."""
    result = {}
    for role, paths in files.items():
        entries = {}
        for name, path in sorted(paths.items()):
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as source:
                for block in iter(lambda: source.read(8*1024*1024), b""):
                    size += len(block)
                    digest.update(block)
            entries[name] = {"size_bytes": size, "sha256": digest.hexdigest()}
        result[role] = {"files": entries, "sha256": _digest_manifest(entries)}
    return result

def validate_artifact_identity(identity):
    if not isinstance(identity, dict) or set(identity) != {"model", "tokenizer", "implementation"}:
        raise ValueError("result lacks model/tokenizer/implementation identities; rerun with the current benchmark entry point")
    for role, item in identity.items():
        if not isinstance(item, dict) or set(item) != {"files", "sha256"}:
            raise ValueError(f"invalid {role} identity")
        files = item["files"]
        if not isinstance(files, dict) or not files:
            raise ValueError(f"empty {role} identity")
        for name, entry in files.items():
            if not isinstance(name, str) or not isinstance(entry, dict) or set(entry) != {"size_bytes", "sha256"}:
                raise ValueError(f"invalid {role} file identity")
            size, digest = entry["size_bytes"], entry["sha256"]
            if type(size) is not int or size < 0 or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"invalid {role} size or SHA256")
        if item["sha256"] != _digest_manifest(files):
            raise ValueError(f"{role} identity digest mismatch")

def validate_hardware_record(profile, hardware):
    from .profiles import validate_hardware, validate_evaluator_version
    required = {"gpu", "capability", "torch", "python", "lm_eval"}
    if not isinstance(hardware, dict) or set(hardware) != required:
        raise ValueError("result lacks a complete hardware/software record")
    for key in required - {"capability"}:
        if not isinstance(hardware[key], str) or not hardware[key]:
            raise ValueError(f"invalid hardware field: {key}")
    validate_hardware(profile, device_name=hardware["gpu"], compute_capability=hardware["capability"], visible_count=1)
    validate_evaluator_version(profile, hardware["lm_eval"])
