"""CPU scoring import checks; no model, dataset, or generated code is executed."""
from types import SimpleNamespace
from importlib.metadata import PackageNotFoundError
import unittest
from unittest.mock import Mock, patch
from benchmarks.task_dependencies import preflight_task_dependencies

class DependencyTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict('benchmarks.task_dependencies.os.environ', {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        version_patch = patch('benchmarks.task_dependencies.version', return_value='0.3.1')
        self.version = version_patch.start()
        self.addCleanup(version_patch.stop)

    def test_missing_dependencies_fail_clearly(self):
        for task, module in [('gsm8k','lm_eval'),('humaneval_plus','evalplus.data')]:
            with patch('benchmarks.task_dependencies.import_module',side_effect=ModuleNotFoundError('synthetic missing dependency')):
                with self.assertRaisesRegex(RuntimeError,module):preflight_task_dependencies(task)

    def test_missing_or_noncallable_interfaces_rejected(self):
        for module in (SimpleNamespace(),SimpleNamespace(simple_evaluate=None)):
            with patch('benchmarks.task_dependencies.import_module',return_value=module):
                with self.assertRaisesRegex(RuntimeError,'unavailable or incompatible'):
                    preflight_task_dependencies('gsm8k')

    def test_present_interfaces_are_not_called(self):
        functions={name:Mock() for name in ('simple_evaluate','load_dataset','get_human_eval_plus','get_human_eval_plus_hash')}
        with patch('benchmarks.task_dependencies.import_module',return_value=SimpleNamespace(**functions)):
            preflight_task_dependencies('gsm8k')
            preflight_task_dependencies('humaneval_plus')
        for function in functions.values():function.assert_not_called()

    def test_unsupported_tasks_fail_before_imports(self):
        with patch('benchmarks.task_dependencies.import_module') as imported:
            for task in ('ifeval','math500','mbpp_plus','unknown'):
                with self.assertRaises(ValueError):preflight_task_dependencies(task)
            imported.assert_not_called()

    def test_evalplus_version_is_checked(self):
        with patch('benchmarks.task_dependencies.import_module', return_value=SimpleNamespace(
                get_human_eval_plus=Mock(), get_human_eval_plus_hash=Mock())):
            preflight_task_dependencies('humaneval_plus')
        self.version.assert_called_once_with('evalplus')

    def test_evalplus_version_drift_fails_before_dataset_import(self):
        for installed in ('0.3.0', '0.3.2', '0.3.1.dev0'):
            self.version.return_value = installed
            with patch('benchmarks.task_dependencies.import_module') as imported:
                with self.assertRaisesRegex(RuntimeError, r'evalplus==0\.3\.1'):
                    preflight_task_dependencies('humaneval_plus')
                imported.assert_not_called()

    def test_missing_evalplus_distribution_fails_clearly(self):
        self.version.side_effect = PackageNotFoundError('evalplus')
        with patch('benchmarks.task_dependencies.import_module') as imported:
            with self.assertRaisesRegex(RuntimeError, r'evalplus==0\.3\.1'):
                preflight_task_dependencies('humaneval_plus')
            imported.assert_not_called()

    def test_evalplus_dataset_override_is_rejected(self):
        for override in ('/tmp/custom-humaneval.jsonl', ''):
            with patch.dict('benchmarks.task_dependencies.os.environ', {'HUMANEVAL_OVERRIDE_PATH': override}):
                with patch('benchmarks.task_dependencies.import_module') as imported:
                    with self.assertRaisesRegex(RuntimeError, 'unset HUMANEVAL_OVERRIDE_PATH'):
                        preflight_task_dependencies('humaneval_plus')
                    imported.assert_not_called()

    def test_gsm8k_does_not_require_evalplus(self):
        with patch('benchmarks.task_dependencies.import_module', return_value=SimpleNamespace(
                simple_evaluate=Mock(), load_dataset=Mock())):
            preflight_task_dependencies('gsm8k')
        self.version.assert_not_called()

    def test_direct_code_evaluation_checks_before_generation(self):
        from benchmarks.generation import evaluate_code
        wrap = Mock()
        with patch('benchmarks.task_dependencies.preflight_task_dependencies',
                   side_effect=RuntimeError('invalid EvalPlus environment')) as preflight:
            with self.assertRaisesRegex(RuntimeError, 'invalid EvalPlus environment'):
                evaluate_code(wrap, 'humaneval_plus', {'dataset': 'humaneval'}, None)
            preflight.assert_called_once_with('humaneval_plus')
        wrap.gen_code.assert_not_called()

if __name__=='__main__':unittest.main()
