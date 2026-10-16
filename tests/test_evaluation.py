import unittest
from benchmarks.profiles import task_protocol
from benchmarks.scoring import _evalplus_status_pass_at_1, _truncate_completion, _extract_acc


class EvaluationTests(unittest.TestCase):
    def test_real_pass_at_one_requires_base_and_plus(self):
        r={"eval":{"a":[{"base_status":"fail","plus_status":"pass"}],
                    "b":[{"base_status":"pass","plus_status":"pass"}]}}
        self.assertEqual(_evalplus_status_pass_at_1(r),(0.5,0.5))

    def test_pass_at_one_is_per_problem(self):
        r={"eval":{"a":[{"base_status":"pass","plus_status":"pass"}]*3,
                    "b":[{"base_status":"fail","plus_status":"fail"}]}}
        self.assertEqual(_evalplus_status_pass_at_1(r),(0.5,0.5))

    def test_stop_first(self):
        self.assertEqual(_truncate_completion("abc\nprint(1)\ndef f():",["\ndef","\nprint"]),"abc")

    def test_missing_metric_does_not_become_zero(self):
        with self.assertRaises(ValueError): _extract_acc({"acc,none":0.5},"gsm8k","exact_match,flexible-extract")
        with self.assertRaises(ValueError): _extract_acc({"exact_match,flexible-extract":float("nan")},"gsm8k","exact_match,flexible-extract")

    def test_unsupported_task_rejected(self):
        for task in ("ifeval","math500","mbpp_plus"):
            with self.assertRaises(ValueError):task_protocol(task)


if __name__=="__main__": unittest.main()
