import pytest

from quantization.optimization.config import (
    DistillationConfig,
    InferenceConfig,
    MixedPrecisionPolicy,
    PrecisionConfig,
    PrecisionMode,
    select_precision,
)


def test_each_precision_mode_is_selectable_without_gpu():
    expected = {
        PrecisionMode.FP16: ("native", True),
        PrecisionMode.BF16: ("native", True),
        PrecisionMode.FP8: ("planned", False),
        PrecisionMode.INT8: ("implemented", True),
        PrecisionMode.INT4: ("planned", False),
    }
    for mode, (status, executable) in expected.items():
        selected = select_precision(PrecisionConfig(mode=mode))
        assert selected.mode is mode
        assert selected.status == status
        assert selected.executable is executable


def test_mixed_precision_module_prefix_override():
    policy = MixedPrecisionPolicy(
        parameter_dtype="float16",
        compute_dtype="float16",
        autocast_dtype="bfloat16",
        module_overrides={"blocks.2": "float32", "blocks.2.attn": "bfloat16"},
    )
    assert policy.dtype_for("blocks.1.attn") == "float16"
    assert policy.dtype_for("blocks.2.ffn") == "float32"
    assert policy.dtype_for("blocks.2.attn.q") == "bfloat16"


def test_four_step_requires_explicit_distillation_artifact():
    with pytest.raises(ValueError, match="distilled model"):
        InferenceConfig(sampling_steps=4).validate()
    config = InferenceConfig(
        sampling_steps=4,
        distillation=DistillationConfig(enabled=True, steps=4, artifact="local://adapter"),
    )
    config.validate()


def test_four_step_config_does_not_claim_backend_execution():
    selection = select_precision(PrecisionConfig(mode=PrecisionMode.FP8))
    assert selection.status == "planned"
    assert not selection.executable


def test_lightning_four_step_settings_are_declared_and_validated():
    config = DistillationConfig(
        enabled=True,
        steps=4,
        artifact="lightx2v/Wan2.2-Lightning",
        method="lora",
    )
    config.validate()
    assert config.solver == "euler"
    assert config.sample_shift == 5.0
    assert config.guide_scale == (1.0, 1.0)


def test_lightning_distillation_rejects_non_euler_sampler():
    config = DistillationConfig(
        enabled=True,
        artifact="lightx2v/Wan2.2-Lightning",
        solver="unipc",
    )
    with pytest.raises(ValueError, match="Euler solver"):
        config.validate()
