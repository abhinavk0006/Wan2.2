from quantization.optimization.model_loading import ExpertModelLoader


def test_model_loader_is_independent_and_records_load_timing():
    loaded_names = []
    loader = ExpertModelLoader(lambda name: loaded_names.append(name) or object())
    result = loader.load("high_noise")
    assert loaded_names == ["high_noise"]
    assert result.name == "high_noise"
    assert result.model is not None
    assert result.measurement.label == "load:high_noise"
    assert result.measurement.elapsed_seconds >= 0
