"""CUDA-free unit checks for the reusable Kaggle runner surface."""

import argparse
import ast
import unittest
from pathlib import Path

from kaggle_generate_video import WanVideoWorker, build_parser


class KaggleRunnerTests(unittest.TestCase):
    def test_module_parses_without_cuda(self):
        source = Path(__file__).with_name("kaggle_generate_video.py").read_text()
        ast.parse(source)

    def test_daemon_flag_and_task_defaults(self):
        args = build_parser().parse_args(["--daemon"])
        self.assertTrue(args.daemon)
        self.assertEqual(args.frames, 17)
        self.assertEqual(args.steps, 20)

    def test_runtime_validation_is_cuda_free(self):
        args = argparse.Namespace(
            lightning=False, model_dir=None, steps=20, frames=17, max_area=1
        )
        worker = WanVideoWorker.__new__(WanVideoWorker)
        worker.args = args
        worker._validate_runtime_args()

    def test_invalid_frame_count_is_rejected(self):
        args = argparse.Namespace(
            lightning=False, model_dir=None, steps=20, frames=18, max_area=1
        )
        worker = WanVideoWorker.__new__(WanVideoWorker)
        worker.args = args
        with self.assertRaises(ValueError):
            worker._validate_runtime_args()

    def test_wrapper_protocol_names_are_documented_in_task_surface(self):
        source = Path(__file__).with_name("kaggle_generate_video.py").read_text()
        for field in ("input_image", "output_video", "clip_duration", "clip_name"):
            self.assertIn(field, source)


if __name__ == "__main__":
    unittest.main()
