"""Ternary training interfaces and CPU quantization regression tests."""
import ast
import argparse
import contextlib
from dataclasses import dataclass, field, fields
import importlib.util
import io
import inspect
import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/bonsai_sft"


def model_arguments(filename):
    """Load only the argument schema, without Trainer or device initialization."""
    tree = ast.parse((PACKAGE / filename).read_text())
    schema = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == "ModelArguments")
    namespace = dict(dataclass=dataclass, field=field, BONSAI_MASK_TOKEN_ID=151654)
    exec(compile(ast.Module(body=[schema], type_ignores=[]), filename, "exec"), namespace)
    return namespace["ModelArguments"]


class TrainingScopeTests(unittest.TestCase):
    def test_native_training_defaults_and_choices_are_ternary(self):
        for filename in ("train.py", "sdlm_train.py"):
            with self.subTest(filename=filename):
                schema = model_arguments(filename)
                args = schema(model_name_or_path="local-model")
                self.assertEqual(args.quant_scheme, "ternary")
                self.assertTrue(args.quantize_embedding)
                quant = next(item for item in fields(schema) if item.name == "quant_scheme")
                self.assertEqual(quant.metadata["choices"], ("ternary",))

    def test_native_training_rejects_other_quantization_before_model_loading(self):
        for filename in ("train.py", "sdlm_train.py"):
            schema = model_arguments(filename)
            for scheme in ("q1_0_g128", "binary", "mixed", "unknown"):
                with self.subTest(filename=filename, scheme=scheme):
                    with self.assertRaisesRegex(ValueError, "requires quant_scheme='ternary'"):
                        schema(model_name_or_path="absent", quant_scheme=scheme)
            with self.assertRaises(ValueError):
                schema(model_name_or_path="absent", quantize_embedding=False)

    def test_recipe_launcher_rejects_other_quantization(self):
        spec = importlib.util.spec_from_file_location("training_launcher", ROOT / "scripts/train_bonsai.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        base = json.loads((ROOT / "configs/training/mtp_example.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "recipe.json"
            args = SimpleNamespace(config=str(config), nproc_per_node=8, nnodes=1,
                                   node_rank=0, model="model", data="data", output="out",
                                   master_addr="localhost", master_port=29500)
            for changes in ({"quant_scheme": "q1_0_g128"}, {"quantize_embedding": False}):
                recipe = dict(base, training_args=dict(base["training_args"], **changes))
                config.write_text(json.dumps(recipe))
                with self.assertRaisesRegex(ValueError, "require ternary QAT"):
                    launcher.command_from_recipe(args)

    def test_binary_export_entry_is_not_packaged(self):
        self.assertFalse((PACKAGE / "export_q1_0.py").exists())

    def test_standalone_eager_entry_redirects_to_validated_cli(self):
        tree = ast.parse((PACKAGE / "gpu/generate.py").read_text())
        self.assertFalse(any(isinstance(node, ast.FunctionDef) and node.name == "_smoke"
                             for node in tree.body))
        entry = tree.body[-1]
        with self.assertRaisesRegex(SystemExit, "bonsai_sft.gpu.run_generation"):
            exec(compile(ast.Module(body=[entry], type_ignores=[]), "generate.py", "exec"),
                 {"__name__": "__main__"})

    def export_cli(self):
        tree = ast.parse((PACKAGE / "export_ternary.py").read_text())
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        export = Mock()
        namespace = dict(argparse=argparse, logging=logging, export_round_to_safetensors=export)
        exec(compile(ast.Module(body=[main], type_ignores=[]), "export_ternary.py", "exec"), namespace)
        return namespace["main"], export

    def test_ternary_export_cli_preserves_hf_export_arguments(self):
        main, export = self.export_cli()
        with patch("sys.argv", ["export_ternary", "--ckpt", "input", "--out", "output"]):
            main()
        export.assert_called_once_with("input", "output")

    def test_ternary_export_cli_rejects_external_conversion_options(self):
        for extra in (["--run_gguf", "true"], ["--llama_cpp", "external"],
                      ["--gguf_type", "Q2_0"]):
            with self.subTest(extra=extra):
                main, export = self.export_cli()
                with patch("sys.argv", ["export_ternary", "--ckpt", "input", "--out", "output", *extra]):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        main()
                self.assertEqual(error.exception.code, 2)
                export.assert_not_called()

    def test_ternary_export_has_no_external_converter(self):
        tree = ast.parse((PACKAGE / "export_ternary.py").read_text())
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(functions, {"validate_sdlm_config", "round_latent_to_ternary",
                                     "export_round_to_safetensors", "main"})
        imports = {name.name for node in tree.body if isinstance(node, ast.Import) for name in node.names}
        self.assertNotIn("subprocess", imports)


try:
    import torch
    import torch.nn.functional as F
    from bonsai_sft import convert, ternary_modules as ternary
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU torch is required")
class TernaryOperatorTests(unittest.TestCase):
    def modeling_constructor(self):
        """Run the constructor with local model-loader stubs and no HF import."""
        tree = ast.parse((PACKAGE / "sdlm_modeling.py").read_text())
        cls = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "BonsaiForSDLM")
        constructor = next(node for node in cls.body
                           if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        cls.body = [constructor]
        config = SimpleNamespace(use_cache=True)
        model = torch.nn.Module()
        model.config = config
        config_loader = Mock(return_value=config)
        model_loader = Mock(return_value=model)
        convert_model = Mock(return_value=(7, 1))
        namespace = dict(nn=torch.nn, torch=torch, os=os, json=json,
                         logger=logging.getLogger(__name__), BONSAI_MASK_TOKEN_ID=151654,
                         AutoConfig=SimpleNamespace(from_pretrained=config_loader),
                         AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader),
                         convert_qwen3_to_ternary_qat=convert_model)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "sdlm_modeling.py", "exec"), namespace)
        return namespace["BonsaiForSDLM"], config_loader, model_loader, convert_model

    def test_modeling_defaults_to_ternary_and_applies_the_supported_converter(self):
        cls, config_loader, model_loader, convert_model = self.modeling_constructor()
        self.assertEqual(inspect.signature(cls).parameters["quant_scheme"].default, "ternary")
        with tempfile.TemporaryDirectory() as directory:
            model = cls(directory)
        config_loader.assert_called_once_with(directory)
        model_loader.assert_called_once()
        convert_model.assert_called_once_with(model.model)
        self.assertFalse(model.model.config.use_cache)
        self.assertEqual(model.block_size, 8)
        self.assertTrue(model.causal_attn)

    def test_modeling_rejects_nonternary_before_reading_config_or_loading_weights(self):
        for kwargs in ({"quant_scheme": "q1_0_g128"}, {"quant_scheme": "binary"},
                       {"quant_scheme": "unknown"}, {"quantize_embedding": False}):
            with self.subTest(kwargs=kwargs):
                cls, config_loader, model_loader, convert_model = self.modeling_constructor()
                with patch("os.path.exists") as exists:
                    with self.assertRaisesRegex(ValueError, "requires quant_scheme='ternary'"):
                        cls("absent", **kwargs)
                exists.assert_not_called()
                config_loader.assert_not_called()
                model_loader.assert_not_called()
                convert_model.assert_not_called()

    def test_package_exports_only_ternary_training_interfaces(self):
        import bonsai_sft
        expected = {"ActQuant", "GROUP_SIZE", "TernaryWeightQuant", "TernaryLinear",
                    "TernaryEmbedding", "freeze_quantize_to_ternary_g128",
                    "convert_qwen3_to_ternary_qat", "count_ternary_modules"}
        self.assertEqual(set(bonsai_sft.__all__), expected)
        for name in expected:
            self.assertTrue(hasattr(bonsai_sft, name), name)
        for name in ("BinaryWeightQuant", "BinaryLinear", "BinaryEmbedding",
                     "convert_qwen3_to_bonsai_qat", "count_binary_modules"):
            self.assertFalse(hasattr(bonsai_sft, name), name)
        for name in ("convert_qwen3_to_bonsai_qat", "count_binary_modules",
                     "_replace_linear", "_replace_embedding"):
            self.assertFalse(hasattr(convert, name), name)

    def test_linear_forward_and_gradients_match_plain_ternary_qat(self):
        for dtype in (torch.float32, torch.bfloat16):
            for bias in (False, True):
                with self.subTest(dtype=dtype, bias=bias):
                    torch.manual_seed(7)
                    layer = ternary.TernaryLinear(256, 4, bias=bias).to(dtype=dtype)
                    x = torch.randn(2, 3, 256, dtype=dtype, requires_grad=True)
                    xr = x.detach().clone().requires_grad_()
                    wr = layer.weight.detach().clone().requires_grad_()
                    br = layer.bias.detach().clone().requires_grad_() if bias else None
                    actual = layer(x)
                    expected = F.linear(ternary.ActQuant.apply(xr),
                                        ternary.TernaryWeightQuant.apply(wr), br)
                    self.assertTrue(torch.equal(actual, expected))
                    actual.float().square().sum().backward()
                    expected.float().square().sum().backward()
                    self.assertTrue(torch.equal(x.grad, xr.grad))
                    self.assertTrue(torch.equal(layer.weight.grad, wr.grad))
                    if bias:
                        self.assertTrue(torch.equal(layer.bias.grad, br.grad))
                    self.assertEqual(set(layer.state_dict()), {"weight", "bias"} if bias else {"weight"})

    def test_embedding_and_export_use_the_same_group128_grid(self):
        torch.manual_seed(8)
        layer = ternary.TernaryEmbedding(5, 256)
        indices = torch.tensor([[0, 4, 2]])
        rounded = ternary.freeze_quantize_to_ternary_g128(layer.weight)
        expected = F.embedding(indices, rounded)
        self.assertTrue(torch.equal(layer(indices), expected))
        self.assertEqual(ternary.GROUP_SIZE, 128)

    def test_experimental_quantization_apis_are_absent(self):
        self.assertFalse(hasattr(convert, "convert_qwen3_to_hetbit_qat"))
        for name in ("set_tequila_eps", "get_tequila_eps", "enable_hetteq", "_TEQUILA_EPS"):
            self.assertFalse(hasattr(ternary, name), name)


if __name__ == "__main__":
    unittest.main()
