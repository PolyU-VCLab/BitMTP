"""Run one Bonsai generation benchmark shard with explicit profile and local model artifacts.

EvalPlus executes model-generated Python: use an isolated container without credentials,
network access, or private mounts. --allow-code-execution acknowledges this requirement;
it does not create a security sandbox.
"""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from .profiles import PROFILES, TASKS, task_protocol, validate_profile, validate_hardware, validate_evaluator_version, validate_bonsai_config
from .contracts import TIMING_CONTRACT, artifact_files, fingerprint_artifacts, implementation_files

def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, default=str, allow_nan=False)
        f.write("\n")

def build_adapter(profile, args, protocol):
    from .bonsai_adapter import BonsaiMTPWrap
    wrap = BonsaiMTPWrap(0, profile, checkpoint=args.checkpoint,
                         tokenizer_path=args.tokenizer, profile_name=args.profile)
    wrap.task_max_gen_tokens = {args.task: protocol["max_gen"]}
    wrap._active_task = args.task
    wrap.init()
    return wrap


def evaluate_code(wrap, task, protocol, output, *, on_generated=None, generate_only=False, started=None):
    if task != "humaneval_plus" or protocol.get("dataset") != "humaneval":
        raise ValueError("code evaluation supports HumanEval+ only")
    from . import scoring
    problems, dataset_hash = scoring.load_humaneval()
    if on_generated is None:
        raise ValueError("HumanEval generation requires a persistence callback before scoring")
    if len(problems) != protocol["expected_samples"]:
        raise RuntimeError("EvalPlus dataset sample count drift")
    stops = scoring.CODE_STOP_SEQUENCES
    sample_path = output / "samples.jsonl"
    with sample_path.open("x", encoding="utf-8") as f:
        for task_id, problem in problems.items():
            prompt = problem["prompt"]
            completion = scoring._truncate_completion(wrap.gen_code(prompt, max_toks=protocol["max_gen"]), stops)
            f.write(json.dumps(dict(task_id=task_id, completion=completion)) + "\n")
    receipt = dict(dataset_hash=dataset_hash, dataset_version=scoring.HUMANEVAL_VERSION,
        evalplus_version=scoring.EVALPLUS_VERSION, sample_sha256=scoring._sha256_file(sample_path))
    generation = on_generated(list(problems), receipt)
    if generate_only:
        return generation
    from .score_humaneval import score_record
    return score_record(output, output, standalone=False, pipeline_started=started)


def persist_generation(wrap, output, base, task_ids, receipt, started):
    """Save runtime facts before any generated Python can run."""
    from .scoring import _sha256_file
    count = len(task_ids)
    if count != base["task_protocol"]["expected_samples"] or wrap._acc_cnt != count:
        raise RuntimeError("request/sample cardinality drift")
    stats = wrap.get_stats()
    stats["raw_output_tokens"] = getattr(wrap, "_raw_output_tokens", getattr(wrap, "_out_tok", None))
    stats["raw_request_tok_s"] = stats["raw_output_tokens"] / stats["total_time_seconds"]
    receipts = wrap._request_receipts
    if len(receipts) != count or sum(r["raw_output_token_count"] for r in receipts) != stats["raw_output_tokens"]:
        raise RuntimeError("raw token receipt cardinality drift")
    write_json(output / "request_receipts.json", [dict(r, task_id=t) for t, r in zip(task_ids, receipts)])
    receipt = dict(receipt, request_receipts_sha256=_sha256_file(output / "request_receipts.json"))
    record = dict(base, status="GENERATED_UNSCORED", observed_samples=count, stats=stats,
        generation_receipt=receipt, generation_wall_time_seconds=time.perf_counter() - started)
    write_json(output / "generation.json", record)
    return record

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True, choices=sorted(PROFILES))
    ap.add_argument("--profile-file", help="JSON profile; must match --profile's family, mode and width")
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--model-id", required=True, help="Identifier for the local checkpoint and tokenizer")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-gen", type=int)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="Check profile/dependencies, artifact paths and available config without CUDA")
    ap.add_argument("--allow-code-execution", action="store_true")
    ap.add_argument("--generate-only", action="store_true", help="HumanEval+: save all generations and runtime statistics without executing generated Python")
    args=ap.parse_args()
    if args.task == "humaneval_plus" and args.max_gen not in (None, 1024):
        ap.error("HumanEval+ requires --max-gen 1024 under the published protocol")
    profile=validate_profile(json.loads(Path(args.profile_file).read_text()) if args.profile_file else PROFILES[args.profile])
    if any(profile[k] != PROFILES[args.profile][k] for k in ("family","model_size","checkpoint_role","reference_model_id","mode","block_size","device_profile","verification","lm_eval_version","replay_widths","segment_ladder")):
        ap.error("profile name and explicit profile disagree")
    protocol=task_protocol(args.task, max_gen=args.max_gen, device_profile=profile["device_profile"])
    if args.generate_only and protocol["kind"] != "evalplus":
        ap.error("--generate-only supports HumanEval+ only")
    if protocol["kind"]=="evalplus" and not args.generate_only and not args.allow_code_execution:
        ap.error("EvalPlus requires --allow-code-execution inside an isolated evaluation container")
    if not Path(args.checkpoint).is_dir():
        ap.error("--checkpoint must be an existing local packed model directory")
    output=Path(args.output)
    if output.exists() and not args.check_only:
        ap.error("output already exists; use a new result directory")
    os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1" if args.offline else "0",
                      HF_DATASETS_OFFLINE="1" if args.offline else "0")
    if args.allow_code_execution and not args.generate_only:
        os.environ["HF_ALLOW_CODE_EVAL"]="1"
    from .task_dependencies import preflight_task_dependencies
    preflight_task_dependencies(args.task)
    from importlib.metadata import version
    validate_evaluator_version(profile, version("lm-eval"))
    config_path=Path(args.checkpoint)/"bonsai_gpu_config.json"
    validate_bonsai_config(profile,json.loads(config_path.read_text()))
    try:
        files = artifact_files(args.checkpoint, args.tokenizer)
    except ValueError as error:
        ap.error(str(error))
    if args.check_only:
        print(json.dumps(dict(status="profile_dependencies_and_artifact_paths_checked",
            profile=profile,task_protocol=protocol,gpu_executed=False,
            weight_tensor_validation="performed by the runtime builder, not this preflight"),indent=2))
        return
    import torch
    import lm_eval
    count=torch.cuda.device_count()
    if count != 1:
        raise RuntimeError(f"benchmark requires one visible GPU, got {count}; check driver/runtime compatibility")
    validate_hardware(profile,device_name=torch.cuda.get_device_name(0),
        compute_capability=torch.cuda.get_device_capability(0),visible_count=count)
    output.mkdir(parents=True)
    identity_started=time.perf_counter()
    files.update(implementation_files())
    artifact_identity=fingerprint_artifacts(files)
    artifact_identity_seconds=time.perf_counter()-identity_started
    setup_start=time.perf_counter()
    wrap=build_adapter(profile,args,protocol)
    warm=wrap._tok.encode("Answer briefly: what is two plus two?",add_special_tokens=True)
    wrap._call(warm,max_gen=24)
    torch.cuda.synchronize()
    wrap.reset_stats()
    setup_seconds=time.perf_counter()-setup_start
    started=time.perf_counter()
    evaluation_receipt=None
    if protocol["kind"] == "evalplus":
        base = dict(schema="bitmtp_generation_v2", task=args.task,
            family=profile["family"], model_id=args.model_id, profile=args.profile,
            profile_contract=profile, checkpoint_role=profile["checkpoint_role"],
            artifact_identity=artifact_identity, artifact_identity_seconds=artifact_identity_seconds,
            task_protocol=protocol, setup_and_warmup_seconds=setup_seconds,
            timing_contract=dict(TIMING_CONTRACT),
            hardware=dict(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                          torch=torch.__version__, python=platform.python_version(), lm_eval=lm_eval.__version__))
        result = evaluate_code(wrap, args.task, protocol, output, generate_only=args.generate_only, started=started,
            on_generated=lambda ids, receipt: persist_generation(wrap, output, base, ids, receipt, started))
        print(json.dumps(dict(status=result["status"], task=args.task,
                              score_percent=result.get("score_percent"), stats=result["stats"]), indent=2))
        return
    if protocol["kind"]=="lm_eval":
        from lm_eval.tasks import TaskManager
        from .scoring import _extract_acc
        raw=lm_eval.simple_evaluate(model=wrap.lm,tasks=[args.task],num_fewshot=protocol["fewshot"],
            batch_size=1,task_manager=TaskManager(),
            log_samples=True,confirm_run_unsafe_code=args.allow_code_execution,
            apply_chat_template=protocol["chat"],fewshot_as_multiturn=protocol["multiturn"],
            random_seed=0,numpy_random_seed=1234,torch_random_seed=1234,fewshot_random_seed=1234)
        write_json(output/"lm_eval_raw.json",raw)
        metrics=raw["results"][args.task]
        score=_extract_acc(metrics,args.task,protocol["metric"])
        count=wrap._acc_cnt
    if count!=protocol["expected_samples"] or wrap._acc_cnt!=count:
        raise RuntimeError("request/sample cardinality drift")
    stats=wrap.get_stats()
    stats["raw_output_tokens"]=getattr(wrap,"_raw_output_tokens",getattr(wrap,"_out_tok",None))
    stats["raw_request_tok_s"]=stats["raw_output_tokens"]/stats["total_time_seconds"]
    receipts=wrap._request_receipts
    if len(receipts)!=count or sum(r["raw_output_token_count"] for r in receipts)!=stats["raw_output_tokens"]:
        raise RuntimeError("raw token receipt cardinality drift")
    write_json(output/"request_receipts.json",receipts)
    result=dict(schema="bitmtp_generation_v2",status="PASS",task=args.task,
        family=profile["family"],model_id=args.model_id,profile=args.profile,profile_contract=profile,checkpoint_role=profile["checkpoint_role"],
        artifact_identity=artifact_identity,artifact_identity_seconds=artifact_identity_seconds,
        task_protocol=protocol,score_percent=score,raw_metrics=metrics,observed_samples=count,stats=stats,
        setup_and_warmup_seconds=setup_seconds,wall_time_seconds=time.perf_counter()-started,
        timing_contract=dict(TIMING_CONTRACT),
        evalplus_receipt=evaluation_receipt,
        hardware=dict(gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),
                      torch=torch.__version__,python=platform.python_version(),lm_eval=lm_eval.__version__))
    write_json(output/"result.json",result)
    print(json.dumps(dict(status="PASS",task=args.task,score_percent=score,stats=stats),indent=2))

if __name__=="__main__":
    main()
