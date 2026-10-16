import dataclasses
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / 'configs/training'
spec = importlib.util.spec_from_file_location('train_bonsai', ROOT / 'scripts/train_bonsai.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class TrainingRecipeTests(unittest.TestCase):
    def recipes(self):
        return [TRAINING / 'sft_example.json', TRAINING / 'mtp_example.json']

    def args(self, path, workers=8):
        return SimpleNamespace(config=str(path), model='model-input', data='prepared.jsonl', output='new-output',
                               nproc_per_node=workers, nnodes=1, node_rank=0,
                               master_addr='localhost', master_port=29500)

    def test_only_two_examples_and_shared_deepspeed_config(self):
        self.assertEqual({p.name for p in TRAINING.glob('*.json')},
                         {'sft_example.json', 'mtp_example.json', 'deepspeed_zero2.json'})

    def test_example_parameters(self):
        common = dict(quant_scheme='ternary', max_length=4096, enable_thinking=False,
                      num_train_epochs=1, per_device_train_batch_size=4, learning_rate=2e-5,
                      weight_decay=0.1, warmup_ratio=0.03, lr_scheduler_type='constant_with_warmup',
                      optim='adamw_torch', bf16=True, tf32=True, gradient_checkpointing=True,
                      save_safetensors=True, logging_steps=10, save_total_limit=3, seed=42,
                      dataloader_num_workers=4, dataloader_pin_memory=True, report_to='none',
                      remove_unused_columns=False, deepspeed='deepspeed_zero2.json', save_steps=500)
        for path, stage in zip(self.recipes(), ('sft', 'mtp')):
            with self.subTest(stage=stage):
                training = dict(common, max_grad_norm=1.0 if stage == 'sft' else 1000.0)
                if stage == 'mtp':
                    training.update(block_size=4, causal_attn=True, mask_token_id=151654,
                                    expected_mask_repeat_times=2)
                self.assertEqual(json.loads(path.read_text()),
                                 dict(stage=stage, model_size='1p7b', reference_world_size=8,
                                      global_batch_size=256, training_args=training))

    def test_example_routes_and_deepspeed(self):
        for path, module in zip(self.recipes(), ('bonsai_sft.train', 'bonsai_sft.sdlm_train')):
            command, cfg = launcher.command_from_recipe(self.args(path))
            self.assertEqual(command[command.index('--module') + 1], module)
            self.assertEqual(cfg['global_batch_size'], 256)
            self.assertEqual(command[command.index('--gradient_accumulation_steps') + 1], '8')
            self.assertEqual(command[command.index('--quant_scheme') + 1], 'ternary')
            self.assertNotIn('--resume_from_checkpoint', command)
            ds_path = Path(command[command.index('--deepspeed') + 1])
            self.assertEqual(ds_path, (TRAINING / 'deepspeed_zero2.json').resolve())
            self.assertEqual(json.loads(ds_path.read_text())['zero_optimization']['stage'], 2)

    def test_effective_arguments_exist_in_training_interfaces(self):
        from bonsai_sft import train, sdlm_train
        for path in self.recipes():
            cfg = json.loads(path.read_text())
            module = train if cfg['stage'] == 'sft' else sdlm_train
            training = module.BonsaiTrainingArguments if cfg['stage'] == 'sft' else module.BonsaiSDLMTrainingArguments
            allowed = {f.name for cls in (module.ModelArguments, module.DataArguments, training)
                       for f in dataclasses.fields(cls)}
            self.assertFalse(set(cfg['training_args']) - allowed)

    def test_global_batch_is_preserved(self):
        for recipe in self.recipes():
            for workers in (1, 2, 4, 8, 16):
                cmd, cfg = launcher.command_from_recipe(self.args(recipe, workers))
                accumulation = int(cmd[cmd.index('--gradient_accumulation_steps') + 1])
                self.assertEqual(accumulation * workers * cfg['training_args']['per_device_train_batch_size'], 256)
            with self.assertRaises(ValueError):
                launcher.command_from_recipe(self.args(recipe, 3))

    def test_mtp_d4_and_d8_settings_parse(self):
        from bonsai_sft import sdlm_train
        from transformers import HfArgumentParser
        config = json.loads((TRAINING / 'mtp_example.json').read_text())
        parser = HfArgumentParser((sdlm_train.ModelArguments, sdlm_train.DataArguments))
        allowed = {field.name for cls in parser.dataclass_types for field in dataclasses.fields(cls)}
        with tempfile.TemporaryDirectory() as directory:
            for block_size in (4, 8):
                modified = dict(config, training_args=dict(config['training_args'], block_size=block_size))
                path = Path(directory) / 'mtp_example.json'
                path.write_text(json.dumps(modified))
                command, _ = launcher.command_from_recipe(self.args(path))
                self.assertEqual(command[command.index('--block_size') + 1], str(block_size))
                selected = {key: value for key, value in modified['training_args'].items() if key in allowed}
                model, data = parser.parse_dict(dict(selected, model_name_or_path='model-input',
                                                    data_path='prepared.jsonl'))
                self.assertEqual(model.block_size, block_size)
                self.assertTrue(model.causal_attn)
                self.assertEqual(model.mask_token_id, 151654)
                self.assertEqual(data.expected_mask_repeat_times, 2)


if __name__ == '__main__':
    unittest.main()
