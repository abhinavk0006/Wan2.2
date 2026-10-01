import time

from quantization.optimization.instrumentation import measure_stage


def test_cpu_stage_instrumentation_measures_time_without_cuda():
    with measure_stage("cpu-smoke") as measurement:
        time.sleep(0.001)
    assert measurement.label == "cpu-smoke"
    assert measurement.device == "cpu"
    assert measurement.elapsed_seconds > 0
    assert measurement.peak_allocated_bytes is None
    assert measurement.to_dict()["elapsed_seconds"] > 0
