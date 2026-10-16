#!/usr/bin/env python3
"""Launch a fresh Bonsai SFT or joint NTP+MTP run from a saved training recipe."""
import argparse,json,subprocess,sys
from pathlib import Path

def command_from_recipe(args):
    config=json.loads(Path(args.config).read_text())
    if config["stage"] not in ("sft","mtp"):raise ValueError("stage must be sft or mtp")
    if args.nproc_per_node<1 or args.nnodes<1 or not 0<=args.node_rank<args.nnodes:
        raise ValueError("invalid distributed worker counts or node rank")
    training=dict(config["training_args"])
    if training.get("quant_scheme") != "ternary" or training.get("quantize_embedding", True) is not True:
        raise ValueError("Bonsai training recipes require ternary QAT with quantized embeddings")
    denominator=training["per_device_train_batch_size"]*args.nproc_per_node*args.nnodes
    if config["global_batch_size"]%denominator:
        raise ValueError("global batch must be divisible by workers times microbatch")
    training.update(gradient_accumulation_steps=config["global_batch_size"]//denominator,
                    model_name_or_path=args.model,data_path=args.data,output_dir=args.output)
    if training.get("deepspeed"):
        training["deepspeed"]=str((Path(args.config).parent/training["deepspeed"]).resolve())
    if any(key in training for key in ("resume_from_checkpoint","save_step_origin","strict_anchored_save_steps")):
        raise ValueError("this portable launcher starts fresh runs; it does not replay a cluster checkpoint/resume chain")
    module="bonsai_sft.train" if config["stage"]=="sft" else "bonsai_sft.sdlm_train"
    command=[sys.executable,"-m","torch.distributed.run",f"--nproc-per-node={args.nproc_per_node}",
             f"--nnodes={args.nnodes}",f"--node-rank={args.node_rank}",f"--master-addr={args.master_addr}",
             f"--master-port={args.master_port}","--module",module]
    for key,value in training.items():
        if value is not None:command += [f"--{key}",str(value)]
    return command,config

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("config","model","data","output"):ap.add_argument("--"+name,required=True)
    ap.add_argument("--nproc-per-node",type=int,default=1)
    ap.add_argument("--nnodes",type=int,default=1)
    ap.add_argument("--node-rank",type=int,default=0)
    ap.add_argument("--master-addr",default="localhost")
    ap.add_argument("--master-port",type=int,default=29500)
    ap.add_argument("--dry-run",action="store_true")
    args=ap.parse_args()
    if Path(args.output).exists():ap.error("use a new output directory; optimizer-state resume is not implicit")
    if not args.dry_run and not Path(args.data).is_file():ap.error("--data must be an existing prepared JSONL file")
    try:command,config=command_from_recipe(args)
    except (KeyError,ValueError) as error:ap.error(str(error))
    print(json.dumps(dict(command=command,global_batch_size=config["global_batch_size"],
        reference_world_size=config["reference_world_size"],requested_world_size=args.nnodes*args.nproc_per_node,
        stage=config["stage"],model_size=config["model_size"]),indent=2),flush=True)
    if not args.dry_run:subprocess.run(command,check=True)

if __name__=="__main__":main()
