"""CUDA-free unit checks for the reusable Kaggle runner surface."""

import argparse
import ast
import io
import unittest
from unittest import mock
from pathlib import Path

import kaggle_generate_video
from kaggle_generate_video import (
    DAEMON_CUDA_OOM_EXIT_CODE,
    WanVideoWorker,
    WanCudaOOMError,
    build_parser,
    cuda_oom_diagnostics,
    normalize_daemon_task,
)


class KaggleRunnerTests(unittest.TestCase):
    def test_module_parses_without_cuda(self):
        source = Path(__file__).with_name("kaggle_generate_video.py").read_text()
        ast.parse(source)

    def test_daemon_flag_and_task_defaults(self):
        args = build_parser().parse_args(["--daemon"])
        self.assertTrue(args.daemon)
        self.assertEqual(args.frames, 17)
        self.assertEqual(args.steps, 20)

    def test_asymmetric_gpu_budget_flags_are_parsed(self):
        args = build_parser().parse_args([
            "--gpu0-memory-gib", "5", "--gpu1-memory-gib", "11",
        ])
        self.assertEqual(args.gpu0_memory_gib, 5)
        self.assertEqual(args.gpu1_memory_gib, 11)

    def test_cuda_oom_diagnostics_are_explicit_without_cuda(self):
        with mock.patch.object(kaggle_generate_video.torch.cuda, "is_available", return_value=False):
            message = cuda_oom_diagnostics(RuntimeError("allocation failed"))
        self.assertIn("CUDA out of memory during Wan2.2 generation", message)
        self.assertIn("allocation failed", message)

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

    def test_task_parsing_matches_wrapper_and_derives_seed(self):
        task = normalize_daemon_task({
            "input_image": "in.png", "output_video": "out.mp4",
            "prompt": "move", "clip_duration": 1.0, "clip_name": "clip-1",
        })
        self.assertEqual(task["frames"], 17)
        self.assertIsInstance(task["seed"], int)
        self.assertEqual(task["image"], "in.png")

    def test_invalid_task_values_fail_before_model_execution(self):
        with self.assertRaisesRegex(ValueError, "frames must be 4n"):
            normalize_daemon_task({
                "input_image": "in.png", "output_video": "out.mp4",
                "prompt": "move", "frames": 18,
            })
        with self.assertRaisesRegex(ValueError, "steps must be an integer"):
            normalize_daemon_task({
                "input_image": "in.png", "output_video": "out.mp4",
                "prompt": "move", "steps": "many",
            })

    def test_daemon_retains_state_until_exit(self):
        class FakeWorker:
            cleanups = 0
            generations = 0

            def __init__(self, args, daemon=False):
                pass

            def generate(self, task):
                type(self).generations += 1

            def cleanup(self):
                type(self).cleanups += 1

        stdin = io.StringIO(
            '{"input_image":"in.png","output_video":"a.mp4","prompt":"a"}\n'
            '{"input_image":"in.png","output_video":"b.mp4","prompt":"b"}\n'
            '{"action":"exit"}\n'
        )
        stdout = io.StringIO()
        with mock.patch.object(kaggle_generate_video, "WanVideoWorker", FakeWorker), \
             mock.patch.object(kaggle_generate_video.sys, "stdin", stdin), \
             mock.patch.object(kaggle_generate_video.sys, "stdout", stdout):
            self.assertEqual(kaggle_generate_video.run_daemon(object()), 0)
        self.assertEqual(FakeWorker.generations, 2)
        self.assertEqual(FakeWorker.cleanups, 1)
        self.assertIn('"action": "shutdown"', stdout.getvalue())

    def test_daemon_supports_shutdown_and_request_correlation(self):
        class FakeWorker:
            def __init__(self, args, daemon=False):
                pass

            def generate(self, task):
                pass

            def cleanup(self):
                pass

        stdin = io.StringIO(
            '{"request_id":"clip-1","input_image":"in.png","output_video":"a.mp4","prompt":"a"}\n'
            '{"request_id":"stop-1","op":"shutdown"}\n'
        )
        stdout = io.StringIO()
        with mock.patch.object(kaggle_generate_video, "WanVideoWorker", FakeWorker), \
             mock.patch.object(kaggle_generate_video.sys, "stdin", stdin), \
             mock.patch.object(kaggle_generate_video.sys, "stdout", stdout):
            self.assertEqual(kaggle_generate_video.run_daemon(object()), 0)
        self.assertIn('"request_id": "clip-1"', stdout.getvalue())
        self.assertIn('"request_id": "stop-1"', stdout.getvalue())

    def test_daemon_marks_cuda_oom_fatal_and_returns_distinct_exit_code(self):
        class FakeWorker:
            def __init__(self, args, daemon=False):
                pass

            def generate(self, task):
                raise WanCudaOOMError("CUDA out of memory during test")

            def cleanup(self):
                pass

        stdin = io.StringIO(
            '{"request_id":"clip-oom","input_image":"in.png",'
            '"output_video":"a.mp4","prompt":"a"}\n'
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(kaggle_generate_video, "WanVideoWorker", FakeWorker), \
             mock.patch.object(kaggle_generate_video.sys, "stdin", stdin), \
             mock.patch.object(kaggle_generate_video.sys, "stdout", stdout), \
             mock.patch.object(kaggle_generate_video.sys, "stderr", stderr):
            self.assertEqual(
                kaggle_generate_video.run_daemon(object()),
                DAEMON_CUDA_OOM_EXIT_CODE,
            )
        self.assertIn('"error_type": "cuda_oom"', stdout.getvalue())
        self.assertIn('"fatal": true', stdout.getvalue())
        self.assertIn('"request_id": "clip-oom"', stdout.getvalue())
        self.assertIn("Wan daemon CUDA OOM", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
