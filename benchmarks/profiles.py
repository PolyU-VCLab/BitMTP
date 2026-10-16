"""Bonsai checkpoint roles and GSM8K/HumanEval+ generation settings."""
from copy import deepcopy
import math

TASKS = {
    "gsm8k": dict(kind="lm_eval",fewshot=4,chat=True,multiturn=True,
                  metric="exact_match,flexible-extract",max_gen=1024,expected_samples=1319),
    "humaneval_plus": dict(kind="evalplus",dataset="humaneval",fewshot=0,chat=False,multiturn=False,
                           metric="pass_at_1_plus",max_gen=1024,expected_samples=164),
}
MODEL_SPECS={
    "1p7b":dict(dim=2048,ffn_dim=6144,n_layers=28,n_heads=16,n_kv_heads=8,head_dim=128,vocab_size=151669),
    "4b":dict(dim=2560,ffn_dim=9728,n_layers=36,n_heads=32,n_kv_heads=8,head_dim=128,vocab_size=151669),
}
PLATFORMS={"h20":dict(device_name="H20",compute_capability=(9,0),lm_eval_version="0.4.9.2"),
           "rtx3090":dict(device_name="RTX 3090",compute_capability=(8,6),lm_eval_version="0.4.11")}
REPLAY_WIDTHS={4:[1,4,5,6,7,8,9],8:[1,8,9,10,12,16,17]}
SEGMENTS=[256,1024,4096]

def task_protocol(task, *, max_gen=None, device_profile="h20"):
    if device_profile not in PLATFORMS:raise ValueError("platform is not included in the published benchmark recipes")
    if task not in TASKS:raise ValueError("published benchmarks support GSM8K and HumanEval+ only")
    p=deepcopy(TASKS[task])
    if max_gen is not None:
        if max_gen not in {1024,p["max_gen"]}:raise ValueError(f"unsupported token cap for {task}: {max_gen}")
        p["max_gen"]=max_gen
    p.update(prompt_token_limit=1024,protocol_id=f"bonsai_{device_profile}_generation_v1",
             lm_eval_version=PLATFORMS[device_profile]["lm_eval_version"],
             stopping="EOS during generation; task stop strings applied after decoding")
    return p

def validate_profile(p):
    required={"family","model_size","checkpoint_role","reference_model_id","device_profile","block_size",
              "mode","threshold","min_chain","draft_projection","verification","lm_eval_version",
              "replay_widths","segment_ladder","temperature"}
    if set(p)!=required:raise ValueError(f"profile fields mismatch: missing={sorted(required-set(p))}, unknown={sorted(set(p)-required)}")
    if p["family"]!="bonsai" or p["model_size"] not in MODEL_SPECS:raise ValueError("unsupported Bonsai model")
    if p["device_profile"] not in PLATFORMS:raise ValueError("platform is not included in the published benchmark recipes")
    mode,role,D=p["mode"],p["checkpoint_role"],p["block_size"]
    if mode not in ("ar","confidence","speculative","hybrid"):raise ValueError("unsupported decoding policy")
    if role=="sft":
        if mode!="ar" or D is not None:raise ValueError("SFT checkpoints provide the separate AR baseline")
        widths=[1]
    elif role=="mtp":
        if D not in ((4,8) if p["model_size"]=="1p7b" else (8,)):raise ValueError("supported MTP checkpoints are 1.7B D4/D8 and 4B D8")
        widths=REPLAY_WIDTHS[D]
    else:raise ValueError("checkpoint_role must be sft or mtp")
    if p["device_profile"]=="rtx3090":
        if p["model_size"]!="1p7b":
            raise ValueError("RTX3090 recipes support only Bonsai-1.7B")
        if role=="mtp" and D==4 and mode!="hybrid":
            raise ValueError("RTX3090 D4 evaluation provides the hybrid route")
    expected_id=f"bonsai_{p['model_size']}_sft" if role=="sft" else f"bonsai_{p['model_size']}_mtp_d{D}"
    if p["reference_model_id"]!=expected_id:raise ValueError("reference checkpoint identity disagrees with its role")
    if p["replay_widths"]!=widths or p["segment_ladder"]!=SEGMENTS:raise ValueError("replay-width or segment schedule differs from the recipe")
    if p["temperature"]!=0:raise ValueError("only greedy decoding is supported")
    if mode in ("confidence","hybrid"):
        value=p["threshold"]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError("confidence threshold must be finite and within [0,1]")
    elif p["threshold"] is not None:
        raise ValueError("AR/Spec recipes do not use a confidence threshold")
    if p["device_profile"]=="rtx3090":
        projections=("native",) if mode=="ar" else ("bf16_raw",)
    else:
        projections=("fp8","native","bf16_raw") if mode in ("confidence","hybrid") else ("native",)
    if p["draft_projection"] not in projections:raise ValueError("unsupported draft projection for this platform and policy")
    expected_chain=(3 if D==4 else 6) if mode=="hybrid" else None
    if p["min_chain"]!=expected_chain:raise ValueError("hybrid requires D4/M3 or D8/M6; other modes do not use min_chain")
    expected_verify="token_id_equality" if mode in ("speculative","hybrid") else "not_used"
    if p["verification"]!=expected_verify:raise ValueError("verifier branches accept only exact token-ID equality")
    if p["lm_eval_version"]!=PLATFORMS[p["device_profile"]]["lm_eval_version"]:
        raise ValueError("lm-eval version does not match the platform recipe")
    return deepcopy(p)

def validate_hardware(p, *, device_name, compute_capability, visible_count):
    validate_profile(p);expected=PLATFORMS[p["device_profile"]]
    if visible_count!=1:raise ValueError("benchmark requires exactly one visible GPU")
    if expected["device_name"] not in device_name or tuple(compute_capability)!=expected["compute_capability"]:
        raise ValueError(f"profile requires one {expected['device_name']} SM{''.join(map(str,expected['compute_capability']))}")

def validate_evaluator_version(p,version):
    if version!=p["lm_eval_version"]:raise ValueError(f"profile requires lm-eval {p['lm_eval_version']}, got {version}")

def validate_bonsai_config(p,config):
    expected=dict(MODEL_SPECS[p["model_size"]],kernel_kind="t2g128")
    if p["checkpoint_role"]=="mtp":expected.update(block_size=p["block_size"],causal_attn=True,mask_token_id=151654)
    for key,value in expected.items():
        if type(config.get(key)) is not type(value) or config[key]!=value:
            raise ValueError(f"Bonsai checkpoint mismatch: {key} must be {value!r}")
    # Packed files do not encode whether weights came from SFT or MTP training.
    # The separate SFT profile therefore requires the caller's matched SFT artifact;
    # accepting its dimensions is not a training-stage or checkpoint-identity proof.

def builtin_profiles():
    profiles={}
    for size in MODEL_SPECS:
        variants=[("sft",None,"ar")]+[("mtp",D,mode) for D in ((4,8) if size=="1p7b" else (8,))
                                     for mode in ("ar","confidence","speculative","hybrid")]
        for role,D,mode in variants:
            stem="sft" if role=="sft" else f"d{D}"
            name=f"bonsai_{size}_h20_{stem}_{mode}"
            profiles[name]=dict(family="bonsai",model_size=size,checkpoint_role=role,
                reference_model_id=f"bonsai_{size}_sft" if role=="sft" else f"bonsai_{size}_mtp_d{D}",
                device_profile="h20",block_size=D,mode=mode,
                threshold=0.9 if mode in ("confidence","hybrid") else None,
                min_chain=(3 if D==4 else 6) if mode=="hybrid" else None,
                draft_projection="fp8" if mode in ("confidence","hybrid") else "native",
                verification="token_id_equality" if mode in ("speculative","hybrid") else "not_used",
                lm_eval_version="0.4.9.2",replay_widths=[1] if role=="sft" else list(REPLAY_WIDTHS[D]),
                segment_ladder=list(SEGMENTS),temperature=0.0)
    for mode in ("ar","confidence","speculative","hybrid"):
        p=deepcopy(profiles[f"bonsai_1p7b_h20_d8_{mode}"])
        p.update(device_profile="rtx3090",lm_eval_version="0.4.11",
                 draft_projection="native" if mode=="ar" else "bf16_raw")
        profiles[f"bonsai_1p7b_rtx3090_d8_{mode}"]=p
    for stem,mode in (("sft","ar"),("d4","hybrid")):
        p=deepcopy(profiles[f"bonsai_1p7b_h20_{stem}_{mode}"])
        p.update(device_profile="rtx3090",lm_eval_version="0.4.11",
                 draft_projection="native" if mode=="ar" else "bf16_raw")
        profiles[f"bonsai_1p7b_rtx3090_{stem}_{mode}"]=p
    return profiles

PROFILES=builtin_profiles()
