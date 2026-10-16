"""Bonsai T2 generation adapter; replay widths and policy come from the profile."""
import os, time, json, hashlib
import torch
from .scoring import _truncate_completion
GEN_MAX_LEN = _MAX_PROMPT_TOKENS = 1024
_BONSAI_EOS_IDS = frozenset([151643, 151645])
def _fmt_optional(value, spec):
    return "n/a" if value is None else format(value, spec)

class BonsaiMTPWrap:
    """Wraps fast_*_generate as an lm_eval ``LM`` instance for one (mode, threshold)."""

    def __init__(self, gpu_id, profile, *, checkpoint, tokenizer_path, profile_name=None):
        from .profiles import validate_profile
        self.profile = validate_profile(profile)
        self.profile_name = profile_name or self.profile["reference_model_id"] + "_" + self.profile["mode"]
        self.gpu_id, self.mode, self.threshold = gpu_id, self.profile["mode"], self.profile["threshold"]
        self.block_size = self.profile["block_size"]
        self.min_chain = self.profile["min_chain"]
        self.replay_widths = tuple(self.profile["replay_widths"])
        self.segment_ladder = tuple(self.profile["segment_ladder"])
        self.checkpoint, self.tokenizer_path = checkpoint, tokenizer_path
        self.task_max_gen_tokens = {}
        self.use_fp8_draft = self.profile["draft_projection"] == "fp8"
        self.use_bf16_draft = self.profile["draft_projection"] == "bf16_raw"
        self.device = torch.device(f"cuda:{gpu_id}")
        self._fn = self._tok = self._lm = None
        self._active_task = None
        self._reset_stats()

    @property
    def label(self):
        return self.profile_name

    @property
    def prefix(self):
        return self.profile_name

    def _reset_stats(self):
        self._pf_tok = self._dc_tok = self._out_tok = 0
        self._pf_t = self._dc_t = self._tot_t = 0.0
        self._ttft_sum = 0.0
        self._acc_cnt = 0
        self._draft_iters = self._drafted = self._accepted_drafts = 0
        self._verifier_bonus = self._committed_tokens = 0
        self._untimed_out_tok = 0
        self._conf_paths = self._verify_paths = 0
        self._prompt_sequence_digests = []
        self._output_sequence_digests = []
        self._request_receipts = []

    def reset_stats(self):
        self._reset_stats()

    def get_stats(self):
        ds = self._dc_tok / self._dc_t if self._dc_t > 0 else 0
        ps = self._pf_tok / self._pf_t if self._pf_t > 0 else 0
        ts = self._out_tok / self._tot_t if self._tot_t > 0 else 0
        accept_rate = (self._accepted_drafts / self._drafted
                       if self._drafted > 0 else None)
        committed = (self._committed_tokens / self._draft_iters
                     if self._draft_iters > 0 else None)
        at = self._out_tok / self._acc_cnt if self._acc_cnt > 0 else 0
        ttft = self._ttft_sum / self._acc_cnt if self._acc_cnt > 0 else 0.0
        prompt_order_sha = hashlib.sha256(
            ("\n".join(self._prompt_sequence_digests) + "\n").encode()
        ).hexdigest()
        output_order_sha = hashlib.sha256(
            ("\n".join(self._output_sequence_digests) + "\n").encode()
        ).hexdigest()
        return {
            "decode_speed": ds, "prefill_speed": ps, "total_speed": ts,
            "model_input_order_sha256": prompt_order_sha,
            "raw_output_order_sha256": output_order_sha,
            "sequence_record_count": len(self._prompt_sequence_digests),
            "avg_accept": committed, "committed_per_iter": committed,
            "accept_rate": accept_rate,
            "draft_iters": self._draft_iters, "drafted_tokens": self._drafted,
            "accepted_drafts": self._accepted_drafts,
            "verifier_bonus_tokens": self._verifier_bonus,
            "n_conf_path": self._conf_paths, "n_verify_path": self._verify_paths,
            "conf_path_frac": (self._conf_paths / (self._conf_paths + self._verify_paths)
                               if self._conf_paths + self._verify_paths else None),
            "gen_tokens": self._out_tok,
            "timed_decode_tokens": self._dc_tok,
            "untimed_output_tokens": self._untimed_out_tok,
            "committed_tokens": self._committed_tokens,
            "prefill_tokens": self._pf_tok,
            "prefill_time_seconds": self._pf_t,
            "decode_time_seconds": self._dc_t,
            "total_time_seconds": self._tot_t,
            "avg_gen_toks": at,
            "ttft": ttft, "ttft_ms": ttft * 1000.0,
        }

    def init(self):
        if self._fn is not None:
            return
        from transformers import AutoTokenizer
        from bonsai_sft.gpu.fast_runtime import GraphAtlas
        from bonsai_sft.gpu.fast_generate import (
            fast_ar_generate,
            fast_confidence_generate,
            fast_speculative_generate,
            fast_hybrid_generate,
        )
        from bonsai_sft.gpu.generate import build_model

        torch.cuda.set_device(self.device)
        self._tok = AutoTokenizer.from_pretrained(
            self.tokenizer_path, trust_remote_code=True, fix_mistral_regex=False)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

        print(f"[GPU{self.gpu_id}|{self.label}] building atlas "
              f"(fp8_draft={self.use_fp8_draft}, "
              f"bf16_draft={self.use_bf16_draft}) ...", flush=True)
        t0 = time.time()
        m = build_model(self.checkpoint, use_kernel=True, device=self.device)
        from .profiles import validate_bonsai_config
        validate_bonsai_config(self.profile, vars(m.args))
        m.prepare_merged_weights(max_positions=4096)
        if self.mode == "ar":
            roles = ("ar",)
        else:
            draft_role = "draft_bf16" if self.use_bf16_draft else "draft_fp8" if self.use_fp8_draft else "draft"
            roles = (draft_role, "ar") if self.mode == "confidence" else (draft_role, "verify", "ar")
        atlas = GraphAtlas(
            m, m.args, self.device, dtype=torch.bfloat16,
            segment_ladder=self.segment_ladder,
            m_set=self.replay_widths,
            roles=roles,
            capture_verbose=False,
        )
        if atlas.segment_ladder != self.segment_ladder:
            raise RuntimeError(f"segment ladder drift: {atlas.segment_ladder}")
        if atlas.m_set != self.replay_widths:
            raise RuntimeError(f"width schedule drift: {atlas.m_set}")
        expected_roles = tuple(roles)
        if tuple(atlas.roles) != expected_roles:
            raise RuntimeError(
                f"graph-role mismatch: {atlas.roles} != {expected_roles}")
        expected_precisions = {
            "ar": "t2", "verify": "t2", "draft": "t2",
            "draft_fp8": "fp8", "draft_bf16": "bf16_raw"}
        for (width, sigma_i, role), graph in atlas.graphs.items():
            if role not in expected_precisions:
                raise RuntimeError(f"unexpected captured graph role: {role}")
            if graph.lm_head_precision != expected_precisions[role]:
                raise RuntimeError(
                    f"projection drift for {role}: {graph.lm_head_precision}")
            shared = atlas.kv_buffers[sigma_i]
            for layer_i, (key_view, value_view) in enumerate(graph.kv_full):
                if key_view.data_ptr() != shared[0, layer_i].data_ptr():
                    raise RuntimeError("key cache is not shared across graph roles")
                if value_view.data_ptr() != shared[1, layer_i].data_ptr():
                    raise RuntimeError("value cache is not shared across graph roles")
        print(f"[GPU{self.gpu_id}|{self.label}] atlas built in {time.time()-t0:.1f}s",
              flush=True)

        if self.mode == "ar":
            self._fn = lambda ids, max_t: fast_ar_generate(
                m, atlas, ids, max_gen_len=max_t, eos_ids=_BONSAI_EOS_IDS)
        elif self.mode == "confidence":
            self._fn = lambda ids, max_t: fast_confidence_generate(
                m, atlas, ids, threshold=self.threshold,
                max_gen_len=max_t, eos_ids=_BONSAI_EOS_IDS,
                use_fp8_draft=self.use_fp8_draft, use_bf16_draft=self.use_bf16_draft)
        elif self.mode == "speculative":
            self._fn = lambda ids, max_t: fast_speculative_generate(
                m, atlas, ids, max_gen_len=max_t, eos_ids=_BONSAI_EOS_IDS,
                use_fp8_draft=self.use_fp8_draft, use_bf16_draft=self.use_bf16_draft,
                rabv_margin=0.0)
        elif self.mode == "hybrid":
            self._fn = lambda ids, max_t: fast_hybrid_generate(
                m, atlas, ids, threshold=self.threshold,
                hybrid_m=self.min_chain, max_gen_len=max_t,
                eos_ids=_BONSAI_EOS_IDS, use_fp8_draft=self.use_fp8_draft,
                use_bf16_draft=self.use_bf16_draft,
                rabv_margin=0.0)
        else:
            raise ValueError(f"unknown mode {self.mode}")

        # Compile fused decode-control kernels during setup, outside timed requests.
        if self.mode in ("speculative", "hybrid"):
            warm_ids = self._tok.encode("1+1=", add_special_tokens=True)
            warm_input = torch.tensor([warm_ids], device=self.device,
                                      dtype=torch.long)
            print(f"[GPU{self.gpu_id}|{self.label}] warming fused decode kernels ...",
                  flush=True)
            self._fn(warm_input, 24)
            torch.cuda.synchronize()
            print(f"[GPU{self.gpu_id}|{self.label}] fused decode kernels ready",
                  flush=True)

        self._build_lm()

    def _call(self, prompt_ids_list, max_gen=GEN_MAX_LEN):
        ids = prompt_ids_list
        if len(ids) > _MAX_PROMPT_TOKENS:
            ids = [ids[0]] + ids[-(_MAX_PROMPT_TOKENS - 1):]
        input_ids = torch.tensor([ids], device=self.device, dtype=torch.long)
        torch.cuda.synchronize()
        out, st = self._fn(input_ids, max_gen)
        torch.cuda.synchronize()
        oids = [int(token) for token in out.tolist()]
        prompt_digest = hashlib.sha256(
            json.dumps([int(token) for token in ids],
                       separators=(",", ":")).encode()).hexdigest()
        output_digest = hashlib.sha256(
            json.dumps(oids, separators=(",", ":")).encode()).hexdigest()
        self._prompt_sequence_digests.append(prompt_digest)
        self._output_sequence_digests.append(output_digest)
        # accumulate stats
        self._pf_tok += st["prefill_tokens"]
        self._pf_t   += st["prefill_time"]
        raw_output_tokens = int(st["decode_tokens"])
        timed_decode_tokens = int(st["timed_decode_tokens"])
        untimed_output_tokens = int(st["untimed_output_tokens"])
        if timed_decode_tokens < 0 or untimed_output_tokens < 0:
            raise RuntimeError("negative output-token accounting")
        if timed_decode_tokens + untimed_output_tokens != raw_output_tokens:
            raise RuntimeError(
                "decode numerator partition mismatch: "
                f"{timed_decode_tokens}+{untimed_output_tokens}!={raw_output_tokens}")
        self._out_tok += raw_output_tokens
        self._dc_tok += timed_decode_tokens
        self._untimed_out_tok += untimed_output_tokens
        self._dc_t   += max(st["decode_time"], 1e-9)
        self._tot_t  += st["total_time"]
        self._ttft_sum += st["ttft"]
        self._acc_cnt  += 1
        self._draft_iters += int(st.get("draft_iters", 0))
        self._drafted += int(st.get("drafted_tokens", 0))
        self._accepted_drafts += int(st.get("accepted_drafts", 0))
        self._verifier_bonus += int(st.get("verifier_bonus_tokens", 0))
        self._committed_tokens += int(st.get("committed_tokens", 0))
        self._conf_paths += int(st.get("n_conf_path", 0))
        self._verify_paths += int(st.get("n_verify_path", 0))
        if self.mode != "ar":
            di = int(st.get("draft_iters", 0))
            drafted = int(st.get("drafted_tokens", 0))
            accepted = int(st.get("accepted_drafts", 0))
            bonus = int(st.get("verifier_bonus_tokens", 0))
            committed_tokens = int(st.get("committed_tokens", 0))
            n_conf = int(st.get("n_conf_path", 0))
            n_verify = int(st.get("n_verify_path", 0))
            if drafted != self.block_size * di:
                raise RuntimeError("request-level drafted-token invariant failed")
            if di != n_conf + n_verify:
                raise RuntimeError("request-level path/iteration invariant failed")
            origin_offset = (
                0 if self.mode == "confidence" else untimed_output_tokens)
            if raw_output_tokens != origin_offset + accepted + bonus:
                raise RuntimeError("request-level output-origin invariant failed")
            if committed_tokens != accepted + bonus:
                raise RuntimeError("request-level committed-token invariant failed")
            if bonus > n_verify or accepted > drafted:
                raise RuntimeError("request-level acceptance bound failed")
            if self.mode == "confidence" and (n_verify != 0 or bonus != 0):
                raise RuntimeError("confidence route used verifier counters")
            if self.mode == "speculative" and n_conf != 0:
                raise RuntimeError("speculative route used confidence counter")
        self._request_receipts.append(dict(
            input_token_ids_count=len(ids), input_token_ids_sha256=prompt_digest,
            raw_output_token_ids=oids, raw_output_token_count=len(oids),
            timed_decode_tokens=timed_decode_tokens, untimed_output_tokens=untimed_output_tokens,
            decode_time_seconds=float(st["decode_time"]),
            total_time_seconds=float(st["total_time"])))
        return oids

    def _build_lm(self):
        from lm_eval.api.model import LM
        outer = self

        class _LM(LM):
            def generate_until(s, reqs):
                results = []
                total = len(reqs)
                _pr = max(1, total // 10)
                for i, req in enumerate(reqs):
                    if i % _pr == 0:
                        st = outer.get_stats()
                        extra = ""
                        if outer._acc_cnt > 0:
                            extra = (f" dec={st['decode_speed']:.0f}"
                                     f" tot={st['total_speed']:.0f}"
                                     f" ttft={st['ttft_ms']:.1f}ms"
                                     f" committed={_fmt_optional(st['avg_accept'], '.2f')}"
                                     f" avg_gen={st['avg_gen_toks']:.0f}")
                        print(f"[GPU{outer.gpu_id}|{outer.label}] {i}/{total}{extra}",
                              flush=True)
                    ctx = req.args[0]
                    kw  = req.args[1] if len(req.args) > 1 else {}
                    stops = kw.get("until", [])
                    requested_max = int(kw.get("max_gen_toks", GEN_MAX_LEN))
                    expected_max = outer.task_max_gen_tokens.get(outer._active_task)
                    if expected_max is None:
                        raise RuntimeError(
                            f"unknown active task {outer._active_task!r}")
                    if requested_max <= 0:
                        raise RuntimeError(
                            f"invalid task max-gen request: {requested_max}")
                    # Apply the selected task-specific protocol consistently,
                    # independent of lm-eval's version-specific defaults.
                    max_t = expected_max
                    ids = outer._tok.encode(ctx, add_special_tokens=True)
                    oids = outer._call(ids, max_gen=max_t)
                    txt = outer._tok.decode(oids, skip_special_tokens=True)
                    txt = _truncate_completion(txt, stops)
                    results.append(txt)
                return results

            def loglikelihood(s, reqs):
                raise NotImplementedError("BonsaiMTPWrap is gen-only")

            def loglikelihood_rolling(s, reqs):
                raise NotImplementedError

            @property
            def tokenizer(s):
                return outer._tok

            @property
            def tokenizer_name(s):
                return self.tokenizer_path

            def apply_chat_template(s, h, add_generation_prompt=True):
                return outer._tok.apply_chat_template(
                    h, tokenize=False, add_generation_prompt=add_generation_prompt)

            @property
            def chat_template(s):
                return outer._tok.chat_template

        _LM.__module__ = __name__
        self._lm = _LM()

    @property
    def lm(self):
        return self._lm

    def gen_code(self, prompt, max_toks=GEN_MAX_LEN):
        ids = self._tok.encode(prompt, add_special_tokens=True)
        oids = self._call(ids, max_gen=max_toks)
        return self._tok.decode(oids, skip_special_tokens=True)

