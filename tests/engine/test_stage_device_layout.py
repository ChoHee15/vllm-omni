"""Regression tests for issue #5003: a stage whose assigned ``devices`` cannot
fit one local replica must fail early in ``build_vllm_config`` with a clear
message, rather than surfacing as an opaque worker-side ``local rank ... out of
bounds`` assertion.

The guard mirrors the runtime device splitter: a stage's ``devices`` lists the
GPUs visible to a single stage *process*, so its count must equal the per-replica
device width (``get_stage_devices_per_replica`` — tensor-parallel width for LLM
stages) or, for a replica pool, ``num_replicas`` times that width. Global
``data_parallel_size`` / ``pipeline_parallel_size`` are intentionally not
multiplied in: data parallelism is expressed across processes (with a possibly
smaller ``data_parallel_size_local`` per node), not within one stage's device
list.
"""

import types
from unittest import mock

import pytest

from vllm_omni.engine import stage_init_utils
from vllm_omni.engine.stage_init_utils import _check_stage_device_layout, build_vllm_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _stage(
    stage_id,
    devices,
    *,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    num_replicas=1,
    stage_type="llm",
):
    """Build a minimal stage config.

    ``tensor_parallel_size`` / ``pipeline_parallel_size`` live on ``engine_args``
    (their production location), which is where both the guard's
    ``get_stage_devices_per_replica`` and the runtime splitter read the
    per-replica width from.
    """
    return types.SimpleNamespace(
        stage_id=stage_id,
        stage_type=stage_type,
        engine_args={
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
        },
        runtime=types.SimpleNamespace(devices=devices, num_replicas=num_replicas),
    )


def test_tp_broadcast_without_devices_fails_early():
    """stage0 gets tensor_parallel_size=4 (broadcast) but only 1 device -> clear error."""
    stage = _stage(0, devices="0", tensor_parallel_size=4)
    with pytest.raises(ValueError) as excinfo:
        _check_stage_device_layout(stage, {"tensor_parallel_size": 4})
    msg = str(excinfo.value)
    # Message names the stage, the mismatch, and the actionable TP workaround.
    assert "Stage 0" in msg
    assert "1 device" in msg and "4" in msg
    assert "--stage-overrides" in msg


def test_tp_broadcast_message_names_tp_only_when_tp_is_the_cause():
    """The #5003 TP-broadcast explanation must appear only when a resolved
    tensor_parallel_size > 1 is the actual cause (P3): a non-TP mismatch should
    get a generic per-stage hint, not misattributed --tensor-parallel-size
    advice."""
    # TP is the cause: tp=4 on a single device -> TP-specific advice.
    tp_stage = _stage(0, devices="0", tensor_parallel_size=4)
    with pytest.raises(ValueError) as tp_exc:
        _check_stage_device_layout(tp_stage, {"tensor_parallel_size": 4})
    assert "--tensor-parallel-size" in str(tp_exc.value)

    # TP is not the cause: tp=1 but 3 devices declared -> generic advice, and the
    # message must not blame --tensor-parallel-size.
    other_stage = _stage(1, devices="0,1,2", tensor_parallel_size=1)
    with pytest.raises(ValueError) as other_exc:
        _check_stage_device_layout(other_stage, {"tensor_parallel_size": 1})
    other_msg = str(other_exc.value)
    assert "Stage 1" in other_msg
    assert "--tensor-parallel-size" not in other_msg
    assert "--stage-overrides" in other_msg


def test_consistent_tp_and_devices_pass():
    """tensor_parallel_size=4 with 4 assigned devices is valid."""
    stage = _stage(0, devices="0,1,2,3", tensor_parallel_size=4)
    _check_stage_device_layout(stage, {"tensor_parallel_size": 4})


def test_single_gpu_stage_passes():
    """A TP=1 stage on a single device (talker/code2wav default) is valid."""
    stage = _stage(1, devices="1", tensor_parallel_size=1)
    _check_stage_device_layout(stage, {"tensor_parallel_size": 1})


def test_missing_devices_is_skipped():
    """No explicit devices -> vLLM assigns them; nothing to validate here."""
    stage = _stage(0, devices=None, tensor_parallel_size=4)
    _check_stage_device_layout(stage, {"tensor_parallel_size": 4})


def test_replica_pool_layout_passes():
    """Pool mode: num_replicas=2 x per-replica width (tp=2) => 4 devices is valid."""
    stage = _stage(0, devices="0,1,2,3", tensor_parallel_size=2, num_replicas=2)
    _check_stage_device_layout(stage, {"tensor_parallel_size": 2})


def test_replica_template_layout_passes():
    """Template mode: a single per-replica width (tp=2) is valid even with
    num_replicas>1 (the splitter scales it across replicas)."""
    stage = _stage(0, devices="0,1", tensor_parallel_size=2, num_replicas=2)
    _check_stage_device_layout(stage, {"tensor_parallel_size": 2})


def test_global_dp_larger_than_local_is_not_rejected():
    """A multi-node layout (global DP > local) must not be rejected: `devices`
    is per-process and the guard keys off the per-replica width (tp), never the
    global data_parallel_size. E.g. tp=1, dp=4, devices="0" is valid."""
    stage = _stage(0, devices="0", tensor_parallel_size=1)
    # Even with a large global data_parallel_size in the engine args, a
    # single-device tp=1 stage is a valid one-local-engine layout.
    _check_stage_device_layout(
        stage,
        {"tensor_parallel_size": 1, "data_parallel_size": 4, "data_parallel_size_local": 1},
    )


def test_diffusion_stage_uses_world_size():
    """A diffusion stage's per-replica width comes from its parallel world size,
    not tensor_parallel_size; a matching device count passes."""
    stage = types.SimpleNamespace(
        stage_id=0,
        stage_type="diffusion",
        engine_args={"parallel_config": {"world_size": 2}},
        runtime=types.SimpleNamespace(devices="0,1", num_replicas=1),
    )
    _check_stage_device_layout(stage, {"tensor_parallel_size": 1})


def test_pipeline_parallel_counts_toward_per_replica_width():
    """An LLM stage's per-replica width is tp * pp: both are intra-engine and
    consume GPUs on the same stage process (the width split_devices_for_replicas
    carves the pool into). A tp=2, pp=2 stage needs 4 devices — 4 passes, and 2
    (tp only) is rejected."""
    ok = _stage(0, devices="0,1,2,3", tensor_parallel_size=2, pipeline_parallel_size=2)
    _check_stage_device_layout(ok, {"tensor_parallel_size": 2, "pipeline_parallel_size": 2})

    # Supplying only tp devices (2) under-provisions the pp dimension -> rejected
    # early instead of surfacing later as a worker-side out-of-bounds assertion.
    bad = _stage(1, devices="0,1", tensor_parallel_size=2, pipeline_parallel_size=2)
    with pytest.raises(ValueError) as excinfo:
        _check_stage_device_layout(bad, {"tensor_parallel_size": 2, "pipeline_parallel_size": 2})
    msg = str(excinfo.value)
    assert "Stage 1" in msg and "needs 4" in msg
    # Not a top-level-TP-broadcast case, so no TP-specific advice.
    assert "--tensor-parallel-size" not in msg


def test_pipeline_parallel_replica_pool():
    """num_replicas pools scale by the full per-replica width (tp * pp):
    num_replicas=2, tp=1, pp=2 => a 4-device pool (or a 2-device template)."""
    pool = _stage(0, devices="0,1,2,3", tensor_parallel_size=1, pipeline_parallel_size=2, num_replicas=2)
    _check_stage_device_layout(pool, {"tensor_parallel_size": 1, "pipeline_parallel_size": 2})

    template = _stage(0, devices="0,1", tensor_parallel_size=1, pipeline_parallel_size=2, num_replicas=2)
    _check_stage_device_layout(template, {"tensor_parallel_size": 1, "pipeline_parallel_size": 2})


def test_build_vllm_config_fails_before_engine_config_on_mismatch():
    """Pin the production wiring: ``build_vllm_config`` must run the device-layout
    guard *before* ``create_engine_config``. If the guard call is dropped or moved
    after engine-config creation, #5003 regresses while the unit tests above stay
    green — so assert here that a mismatched stage raises and that neither
    ``create_engine_config`` nor ``Executor.get_class`` is reached."""
    stage = _stage(0, devices="0", tensor_parallel_size=4)
    with (
        mock.patch.object(stage_init_utils.OmniEngineArgs, "create_engine_config") as create_engine_config,
        mock.patch.object(stage_init_utils.Executor, "get_class") as get_class,
    ):
        with pytest.raises(ValueError) as excinfo:
            build_vllm_config(
                stage,
                model="dummy-model",
                engine_args_dict={
                    "tensor_parallel_size": 4,
                    "data_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                },
            )
    msg = str(excinfo.value)
    assert "Stage 0" in msg
    assert "--stage-overrides" in msg
    # Guard fired early: worker/config construction was never reached.
    create_engine_config.assert_not_called()
    get_class.assert_not_called()


def test_build_vllm_config_proceeds_on_consistent_layout():
    """The guard must not false-positive: a consistent single-GPU stage flows past
    it and reaches ``create_engine_config`` / ``Executor.get_class`` as usual."""
    stage = _stage(1, devices="1", tensor_parallel_size=1)
    fake_config = types.SimpleNamespace(
        quant_config=None,
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace()),
    )
    sentinel_executor = object()
    with (
        mock.patch.object(
            stage_init_utils.OmniEngineArgs, "create_engine_config", return_value=fake_config
        ) as create_engine_config,
        mock.patch.object(stage_init_utils.Executor, "get_class", return_value=sentinel_executor),
        mock.patch.object(stage_init_utils.OmniINCConfig, "maybe_upgrade", side_effect=lambda quant: quant),
    ):
        vllm_config, executor_class = build_vllm_config(
            stage,
            model="dummy-model",
            engine_args_dict={"tensor_parallel_size": 1},
        )
    create_engine_config.assert_called_once()
    assert vllm_config is fake_config
    assert executor_class is sentinel_executor
