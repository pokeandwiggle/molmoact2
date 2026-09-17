"""The normalizer's clip switch: bounded modes saturate at [-1, 1] only when asked."""

import numpy as np
import pytest

from olmo.data.robot_processing import RobotProcessorConfig, _FeatureNormalizer

# One channel fitted on a q01/q99 band of [-1.0, 1.0] units, whose recorded extremes
# reach twice as far; the second channel is the gripper-style pass-through.
_STATS = {
    "min": [-2.0, 0.0],
    "max": [2.0, 1.0],
    "mean": [0.0, 0.5],
    "std": [0.5, 0.3],
    "q01": [-1.0, 0.0],
    "q99": [1.0, 1.0],
    "mask": [True, False],
}


def _normalizer(clip: bool) -> _FeatureNormalizer:
    normalizer = _FeatureNormalizer.from_stats(_STATS, mode="q01_q99", clip=clip)
    assert normalizer is not None
    return normalizer


def test_clipping_stays_the_default():
    assert _FeatureNormalizer.from_stats(_STATS, mode="q01_q99").clip is True
    assert RobotProcessorConfig().norm_clip is True


def test_a_clipped_normalizer_saturates_the_tails_both_ways():
    normalizer = _normalizer(clip=True)
    normed = normalizer.normalize(np.array([[-2.0, 0.25]]))
    np.testing.assert_allclose(normed, [[-1.0, 0.25]])
    # A model output beyond the band comes back at the band's edge, not beyond it.
    unnormed = normalizer.unnormalize(np.array([[1.7, 0.25]]))
    np.testing.assert_allclose(unnormed, [[1.0, 0.25]])


def test_an_unclipped_normalizer_keeps_the_tails_proportional():
    normalizer = _normalizer(clip=False)
    recorded = np.array([[-2.0, 0.25], [0.5, 0.75], [2.0, 1.0]])
    normed = normalizer.normalize(recorded)
    np.testing.assert_allclose(normed[:, 0], [-2.0, 0.5, 2.0])
    # The masked channel is untouched either way.
    np.testing.assert_allclose(normed[:, 1], recorded[:, 1])
    np.testing.assert_allclose(normalizer.unnormalize(normed), recorded)


@pytest.mark.parametrize("mode", ["mean_std", "none"])
def test_the_switch_is_inert_for_unbounded_modes(mode):
    clipped = _FeatureNormalizer.from_stats(_STATS, mode=mode, clip=True)
    unclipped = _FeatureNormalizer.from_stats(_STATS, mode=mode, clip=False)
    values = np.array([[-3.0, 0.5]])
    np.testing.assert_allclose(clipped.normalize(values), unclipped.normalize(values))


def test_the_config_threads_the_switch_into_every_normalizer():
    config = RobotProcessorConfig.from_stats(
        stats_by_tag={"paw": {"action": _STATS, "observation.state": _STATS}},
        tag_metadata={"paw": {"action_key": "action", "state_keys": ["observation.state"]}},
        repo_to_tag={"pokeandwiggle/demo": "paw"},
        norm_mode="q01_q99",
        norm_clip=False,
    )
    processor = config.build_processor()
    assert processor.action_normalizers["paw"].clip is False
    assert processor.state_normalizers["paw"].clip is False
    normed = processor.normalize_action(np.array([[-2.0, 0.25]]), repo_id="pokeandwiggle/demo")
    np.testing.assert_allclose(normed, [[-2.0, 0.25]])


def test_a_config_written_before_the_switch_still_clips():
    # ``norm_clip`` absent from a checkpoint's stored config means the run trained
    # with the clip, so loading it must keep clipping.
    config = RobotProcessorConfig(
        metadata_by_tag={
            "paw": {
                "action_key": "action",
                "state_keys": ["observation.state"],
                "action_stats": _STATS,
                "state_stats": _STATS,
            }
        },
        norm_mode="q01_q99",
    )
    processor = config.build_processor()
    assert processor.action_normalizers["paw"].clip is True
    normed = processor.normalize_action(np.array([[-2.0, 0.25]]), repo_id="paw")
    np.testing.assert_allclose(normed, [[-1.0, 0.25]])
