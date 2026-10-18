"""CPU numerical checks; no model downloads or CUDA required."""
import importlib.util
import types
import unittest
import tempfile
import json
import time
from unittest import mock
from pathlib import Path

import torch
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OptimizationTests(unittest.TestCase):
    def test_stage_metrics_record_explicit_cumulative_memory_scope(self):
        module = load("profile_worker_under_test", "quantization/kaggle_generate_video.py")
        worker = module.WanVideoWorker.__new__(module.WanVideoWorker)
        worker.args = types.SimpleNamespace(profile_stages=True)
        worker._current_metrics = {"phases": []}
        worker._phase_starts = {}
        worker._request_started = time.perf_counter()
        with mock.patch.object(torch.cuda, "device_count", return_value=0):
            worker.profile_event("vae_decode", "start")
            worker.profile_event("vae_decode", "end")
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "clip.mp4"
                worker._finish_metrics(types.SimpleNamespace(output=output), "success")
                data = json.loads(output.with_suffix(".metrics.json").read_text())
                worker._current_metrics = {"phases": []}
                worker._phase_starts = {"vae_decode": time.perf_counter()}
                worker._finish_metrics(types.SimpleNamespace(output=output), "cuda_oom", "CUDA out of memory")
                failed = json.loads(output.with_suffix(".metrics.json").read_text())
                self.assertEqual(failed["incomplete_stages"], ["vae_decode"])
                self.assertEqual(len(list(Path(directory).glob("clip.metrics.previous-*.json"))), 1)
        self.assertEqual(data["phases"][0]["peak_scope"], "cumulative_since_request_start")
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["incomplete_stages"], [])

    def test_int8_direct_expansion_matches_float32_reference(self):
        cls = load("int8_under_test", "quantization/int8_linear.py").Int8Linear
        torch.manual_seed(42)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            layer = cls.from_linear(torch.nn.Linear(37, 19).to(dtype))
            x = torch.randn(2, 7, 37).to(dtype)
            expected = torch.nn.functional.linear(x, layer.dequantize_weight().to(dtype), layer.bias)
            torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)

    def test_sdpa_respects_padding_scale_and_half_dtype(self):
        module = load("attention_under_test", "wan/modules/attention.py")
        module.FLASH_ATTN_2_AVAILABLE = module.FLASH_ATTN_3_AVAILABLE = False
        torch.manual_seed(42)
        q, k, v = [torch.randn(2, 5, 2, 8).half() for _ in range(3)]
        output = module.attention(q, k, v, q_lens=torch.tensor([3, 5]),
            k_lens=torch.tensor([2, 4]), q_scale=.75, softmax_scale=.2)
        for i, (nq, nk) in enumerate([(3, 2), (5, 4)]):
            expected = torch.nn.functional.scaled_dot_product_attention(
                q[i:i+1, :nq].transpose(1, 2) * .75,
                k[i:i+1, :nk].transpose(1, 2), v[i:i+1, :nk].transpose(1, 2), scale=.2)
            torch.testing.assert_close(output[i:i+1, :nq], expected.transpose(1, 2))
            self.assertEqual(output.dtype, torch.float16)
        self.assertEqual(torch.count_nonzero(output[0, 3:]), 0)

    def test_prompt_cache_skips_negative_cfg1_and_refreshes_changed_text(self):
        module = load("worker_under_test", "quantization/kaggle_generate_video.py")
        worker = module.WanVideoWorker.__new__(module.WanVideoWorker)
        calls = []
        class Encoder:
            model = object()
            def __call__(self, texts, device):
                calls.append(texts[0])
                return [torch.ones(1, 2)]
        encoder = Encoder()
        worker.model = types.SimpleNamespace(text_encoder=encoder, sample_neg_prompt="default negative")
        worker.args = types.SimpleNamespace(lightning=True)
        worker._prompt_context_cache = {}
        worker._prompt_text_cache = {}
        worker._restore_t5_after_expert_swap = lambda: setattr(encoder, "model", object())
        item = {"cache_key": "paper", "prompt": "paper turns blue", "negative_prompt": "extra tube"}
        worker.prepare_prompts([item])
        self.assertEqual(calls, ["paper turns blue"])
        worker.prepare_prompts([item])
        self.assertEqual(len(calls), 1)
        worker.prepare_prompts([{**item, "prompt": "paper remains red"}])
        self.assertEqual(calls[-1], "paper remains red")
        self.assertEqual(worker._prompt_context_cache["paper"][1], [])

    def test_base_mode_encodes_shared_negative_only_once(self):
        module = load("base_worker_under_test", "quantization/kaggle_generate_video.py")
        worker = module.WanVideoWorker.__new__(module.WanVideoWorker)
        calls = []
        class Encoder:
            model = object()
            def __call__(self, texts, device):
                calls.append(texts[0])
                return [torch.ones(1, 2)]
        worker.model = types.SimpleNamespace(text_encoder=Encoder(), sample_neg_prompt="default negative")
        worker.args = types.SimpleNamespace(lightning=False)
        worker._prompt_context_cache, worker._prompt_text_cache = {}, {}
        worker.prepare_prompts([{"cache_key": "a", "prompt": "first"}, {"cache_key": "b", "prompt": "second"}])
        self.assertEqual(calls.count("default negative"), 1)
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
