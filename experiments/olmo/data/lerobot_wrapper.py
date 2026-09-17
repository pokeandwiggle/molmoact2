"""
Utilities for exposing LeRobot datasets through Molmo's dataset interface.

Dataset names follow the pattern `lerobot:<username>/<repo_id>[@episodes]`.
Episodes can be specified via comma separated values or ranges (e.g.,
`lerobot:myuser/mydataset@0-10,12`). The optional environment variable
`LEROBOT_DATA_ROOT` can be used to point to a directory containing the dataset;
if set, the loader will expect the dataset to live under
`${LEROBOT_DATA_ROOT}/{username}/{repo_id}` (unless the env var already points
directly at the dataset root).

Set the environment variables `LEROBOT_N_OBS_STEPS` and
`LEROBOT_MAX_ACTION_HORIZON` to control how many previous observations and the
global maximum number of future actions are returned for each example when not
passed explicitly to `build_lerobot_dataset(...)`. Per-tag `action_horizon` and
`n_action_steps` are read from `LEROBOT_TAG_METADATA`. Use
`LEROBOT_VIDEO_BACKEND` to choose the LeRobot video backend. The default is
`pyav`; set `LEROBOT_VIDEO_BACKEND=torchcodec` to explicitly use TorchCodec.
Set `LEROBOT_TOLERANCE_S` (default 1e-4) to relax video timestamp tolerance.
Set `LEROBOT_STATS_BY_TAG`/`LEROBOT_REPO_TO_TAG` to share
normalization stats across multiple datasets in the same tag.
Large payloads can also be passed via `LEROBOT_STATS_BY_TAG_PATH`,
`LEROBOT_REPO_TO_TAG_PATH`, and `LEROBOT_TAG_METADATA_PATH`.
Set `LEROBOT_ACTION_FORMAT` to one of {"continuous", "discrete", "both"}:
- continuous: prompt-only text + continuous action flow targets
- discrete: question/answer text where answer is tokenized action string
- both: use both continuous flow targets and discrete answer tokens
Set `LEROBOT_DISCRETE_ACTION_TOKENIZER` to choose the action tokenizer
(default: "physical-intelligence/fast").
Set `LEROBOT_STATE_FORMAT` to one of {"continuous", "discrete", "both"}:
- continuous: pass normalized continuous state vectors to the action expert only
- discrete: serialize normalized state vectors into tokenizer input only
- both: use both continuous action-expert state and discrete state tokens in the input
"""

import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
import time
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Callable

import datasets
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizerFast

from olmo.util import get_hf_access_token

_LEROBOT_SRC = Path(__file__).resolve().parents[2] / "lerobot" / "src"
if _LEROBOT_SRC.is_dir() and str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))

try:
    import av
except Exception:  # pragma: no cover - optional dependency surface
    av = None

from lerobot.datasets import video_utils as lerobot_video_utils
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    hf_transform_to_torch,
    load_nested_dataset,
)

# ---------------------------------------------------------------------------
# Monkey-patch: lerobot's get_safe_version raises RevisionNotFoundError with
# only a message string, but newer huggingface_hub requires a `response` kwarg.
# Return "main" on failure so the dataset loads from the local cache without
# triggering a full re-download via the retry-with-main-revision path.
# ---------------------------------------------------------------------------
import lerobot.datasets.utils as _lerobot_utils

_orig_get_safe_version = _lerobot_utils.get_safe_version


def _patched_get_safe_version(*args, **kwargs):
    try:
        return _orig_get_safe_version(*args, **kwargs)
    except TypeError as _exc:
        if "response" in str(_exc):
            return "main"
        raise


_lerobot_utils.get_safe_version = _patched_get_safe_version
try:
    import lerobot.datasets.lerobot_dataset as _ld_mod
    _ld_mod.get_safe_version = _patched_get_safe_version
except Exception:
    pass

from olmo.extra_tokens import (
    DEFAULT_NUM_DEPTH_TOKENS,
    DEFAULT_NUM_STATE_TOKENS,
    ROBOT_OUTPUT_STYLES,
    SUPPORTED_STATE_FORMATS,
    append_discrete_state_to_prompt,
    build_discrete_action_string_from_action,
    build_discrete_depth_string,
    build_robot_prompt_fields,
    build_discrete_state_string,
    style_uses_action_output,
    style_uses_depth_output,
    wrap_control_text,
    wrap_setup_text,
)
from olmo.data.robot_processing import (
    RobotProcessor,
    RobotProcessorConfig,
)

from olmo.data.dataset import Dataset
from olmo.preprocessing.sequence_length_utils import MalformedExampleError

log = logging.getLogger(__name__)


Number = Union[int, np.integer]
ACTION_TOKENIZER_MAX_ACTION_DIM = 32
SUPPORTED_ACTION_FORMATS = {"continuous", "discrete", "both"}
SUPPORTED_RANDOM_CAMERA_ORDER_MODES = {"none", "episode", "all"}
_IGNORED_LEROBOT_FEATURE_DTYPES = {"audio", "audio_"}
_ACTION_DISCRETE_PROCESSOR_CACHE: Dict[Tuple[str, bool, Optional[str]], Any] = {}
_ANNOTATED_TASK_WARNING_KEYS: set[Tuple[str, str]] = set()
_ANNOTATED_TASKS_CACHE: Dict[Tuple[str, str], Optional[Dict[int, Any]]] = {}
_TASK_TO_EPISODE_CACHE: Dict[Tuple[str, str], Optional[Dict[int, np.ndarray]]] = {}
_FRAME_COUNT_MISMATCH_WARNING_KEYS: set[Tuple[str, str, int, int]] = set()
_QUESTION_TRAILING_SENTENCE_PUNCTUATION = ".,!?;:,…"
_QUESTION_TRAILING_CLOSERS = "\"'”’)]}"
_QUESTION_SURROUNDING_DELIMITERS = "\"'`“”‘’[](){}"
_ANNOTATED_TASKS_FILENAME = "tasks_annotated.parquet"
_ANNOTATED_TASKS_INDEX_NAME = "episode_index"
_ANNOTATED_TASKS_COLUMN_NAME = "task"
_TASK_TO_EPISODE_FILENAME = "task_to_episode.parquet"
_TASK_TO_EPISODE_TASK_INDEX_COLUMN = "task_index"
_TASK_TO_EPISODE_EPISODE_INDEX_COLUMN = "episode_index"
_FROZEN_TASK_SENTINEL = "<ALL FROZEN FRAMES>"
_DEPTH_DATASET_SUFFIX = "_depth"
_QUESTION_PREFIX_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
        r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+",
    )
)


def _should_retry_lerobot_torchcodec_packet_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "could not push packet to decoder" in message
        or "invalid data found when processing input" in message
        or "avcodec_send_packet" in message
        or "no frame!" in message
    )


def _install_lerobot_torchcodec_retry_patch() -> None:
    if getattr(lerobot_video_utils, "_molmo_torchcodec_retry_patch_installed", False):
        return

    cache_cls = lerobot_video_utils.VideoDecoderCache

    def _patched_remove(self, video_path: str) -> None:
        video_path = str(video_path)
        with self._lock:
            cached = self._cache.pop(video_path, None)
        if cached is None:
            return
        _, file_handle = cached
        try:
            file_handle.close()
        except Exception:
            pass

    def _patched_get_decoder(self, video_path: str):
        if importlib.util.find_spec("torchcodec"):
            from torchcodec.decoders import VideoDecoder
        else:
            raise ImportError("torchcodec is required but not available.")

        video_path = str(video_path)
        with self._lock:
            if video_path not in self._cache:
                file_handle = lerobot_video_utils.fsspec.open(video_path).__enter__()
                decoder = VideoDecoder(file_handle, seek_mode="approximate", num_ffmpeg_threads=1)
                self._cache[video_path] = (decoder, file_handle)
            return self._cache[video_path][0]

    def _patched_decode_video_frames_torchcodec(
        video_path: Path | str,
        timestamps: list[float],
        tolerance_s: float,
        log_loaded_timestamps: bool = False,
        decoder_cache: Optional[object] = None,
    ) -> torch.Tensor:
        if decoder_cache is None:
            decoder_cache = lerobot_video_utils._default_decoder_cache

        video_path = str(video_path)
        last_error: Optional[Exception] = None
        frames_batch = None
        for attempt in range(2):
            decoder = decoder_cache.get_decoder(video_path)
            metadata = decoder.metadata
            average_fps = metadata.average_fps
            frame_indices = [round(ts * average_fps) for ts in timestamps]
            try:
                frames_batch = decoder.get_frames_at(indices=frame_indices)
                break
            except Exception as exc:
                last_error = exc
                if attempt == 0 and _should_retry_lerobot_torchcodec_packet_error(exc):
                    log.warning(
                        "TorchCodec decode failed for %s with %s; evicting cached decoder and retrying once.",
                        video_path,
                        exc,
                    )
                    if hasattr(decoder_cache, "remove"):
                        decoder_cache.remove(video_path)
                    continue
                raise

        if frames_batch is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError(f"TorchCodec did not return any frames for {video_path}.")

        loaded_ts: List[float] = []
        loaded_frames: List[torch.Tensor] = []
        for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=True):
            loaded_frames.append(frame)
            loaded_ts.append(pts.item())
            if log_loaded_timestamps:
                log.info("Frame loaded at timestamp=%0.4f", pts)

        query_ts = torch.tensor(timestamps)
        loaded_ts_tensor = torch.tensor(loaded_ts)

        dist = torch.cdist(query_ts[:, None], loaded_ts_tensor[:, None], p=1)
        min_, argmin_ = dist.min(1)

        is_within_tol = min_ < tolerance_s
        if not is_within_tol.all():
            raise lerobot_video_utils.FrameTimestampError(
                f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
                " It means that the closest frame that can be loaded from the video is too far away in time."
                " This might be due to synchronization issues with timestamps during data collection."
                " To be safe, we advise to ignore this item during training."
                f"\nqueried timestamps: {query_ts}"
                f"\nloaded timestamps: {loaded_ts_tensor}"
                f"\nvideo: {video_path}"
            )

        closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
        closest_ts = loaded_ts_tensor[argmin_]

        if log_loaded_timestamps:
            log.info("%s", f"{closest_ts=}")

        closest_frames = (closest_frames / 255.0).type(torch.float32)

        if len(timestamps) != len(closest_frames):
            raise lerobot_video_utils.FrameTimestampError(
                f"Retrieved timestamps differ from queried {set(closest_frames) - set(timestamps)}"
            )

        return closest_frames

    cache_cls.remove = _patched_remove
    cache_cls.get_decoder = _patched_get_decoder
    lerobot_video_utils.decode_video_frames_torchcodec = _patched_decode_video_frames_torchcodec
    lerobot_video_utils._molmo_torchcodec_retry_patch_installed = True


_install_lerobot_torchcodec_retry_patch()


def _hash_hf_token(hf_token: Optional[str]) -> Optional[str]:
    if not hf_token:
        return None
    return hashlib.sha256(hf_token.encode("utf-8")).hexdigest()


def _action_discrete_processor_cache_key(
    discrete_action_tokenizer: str,
    *,
    offline_mode: bool,
    hf_token: Optional[str],
) -> Tuple[str, bool, Optional[str]]:
    return (str(discrete_action_tokenizer), bool(offline_mode), _hash_hf_token(hf_token))


def _reset_action_discrete_processor_cache() -> None:
    _ACTION_DISCRETE_PROCESSOR_CACHE.clear()


def _annotated_tasks_cache_key(repo_id: str, annotated_path: Path) -> Tuple[str, str]:
    try:
        cache_path = str(annotated_path.resolve(strict=False))
    except Exception:
        cache_path = str(annotated_path)
    return (str(repo_id), cache_path)


def _reset_annotated_tasks_cache() -> None:
    _ANNOTATED_TASKS_CACHE.clear()
    _TASK_TO_EPISODE_CACHE.clear()
    _ANNOTATED_TASK_WARNING_KEYS.clear()


def _warn_annotated_task(
    repo_id: str,
    warning_kind: str,
    message: str,
    *args: object,
) -> None:
    dedupe_key = (str(repo_id), warning_kind)
    if dedupe_key in _ANNOTATED_TASK_WARNING_KEYS:
        return
    _ANNOTATED_TASK_WARNING_KEYS.add(dedupe_key)
    log.warning(message, *args)


def _warn_frame_count_mismatch(
    repo_id: str,
    data_root: Path,
    loaded_rows: int,
    expected_rows: int,
) -> None:
    # DataLoader workers and nonzero distributed ranks can all reopen the same
    # dataset independently. Keep this warning on the main rank-0 process so it
    # stays actionable without flooding logs.
    if torch.utils.data.get_worker_info() is not None:
        return
    if _get_dist_rank() != 0:
        return
    try:
        cache_path = str(data_root.resolve(strict=False))
    except Exception:
        cache_path = str(data_root)
    dedupe_key = (str(repo_id), cache_path, int(loaded_rows), int(expected_rows))
    if dedupe_key in _FRAME_COUNT_MISMATCH_WARNING_KEYS:
        return
    _FRAME_COUNT_MISMATCH_WARNING_KEYS.add(dedupe_key)
    log.warning(
        "LeRobot dataset '%s' loaded %d frame rows from %s, but metadata expects %d. "
        "MolmoAct2 will sample from the loaded row count to avoid out-of-bounds indices.",
        repo_id,
        int(loaded_rows),
        data_root,
        int(expected_rows),
    )


def _sample_rng_index(rng: np.random.Generator, high: int) -> int:
    if high <= 0:
        raise ValueError(f"Expected high > 0 when sampling, got {high}.")
    if hasattr(rng, "integers"):
        return int(rng.integers(high))
    if hasattr(rng, "randint"):
        return int(rng.randint(0, high))
    raise TypeError(f"Unsupported RNG object {type(rng)!r}: missing integers()/randint().")


def _normalize_question_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""
    previous = None
    while normalized and normalized != previous:
        previous = normalized
        normalized = normalized.strip().strip(_QUESTION_SURROUNDING_DELIMITERS).strip()
        for pattern in _QUESTION_PREFIX_PATTERNS:
            normalized = pattern.sub("", normalized, count=1).strip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_SENTENCE_PUNCTUATION).rstrip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_CLOSERS).rstrip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_SENTENCE_PUNCTUATION).rstrip()
    sentence_chunks = [chunk.strip() for chunk in re.split(r"[.!?]+", normalized) if chunk.strip()]
    if len(sentence_chunks) > 1:
        normalized = "; ".join(sentence_chunks)
    normalized = normalized.lower()
    return normalized


@dataclass
class _RepoSpec:
    repo_id: str
    episodes: Optional[List[int]] = None


def _filter_hf_loadable_features(
    features: Dict[str, Dict[str, object]],
    *,
    repo_id: str,
) -> Dict[str, Dict[str, object]]:
    filtered: Dict[str, Dict[str, object]] = {}
    ignored: Dict[str, str] = {}
    for key, feature in features.items():
        dtype = str(feature.get("dtype") or "")
        if dtype in _IGNORED_LEROBOT_FEATURE_DTYPES:
            ignored[str(key)] = dtype
            continue
        filtered[str(key)] = feature
    if ignored:
        log.info("Ignoring unsupported LeRobot feature dtypes for %s: %s", repo_id, ignored)
    return filtered


def _get_hf_features_from_features_compatible(features: Dict[str, Dict[str, object]]) -> datasets.Features:
    """Version of LeRobot's feature conversion that also accepts scalar shape [] / ()."""
    hf_features: Dict[str, object] = {}
    for key, ft in features.items():
        dtype = str(ft.get("dtype"))
        shape = ft.get("shape")
        if shape is None:
            raise ValueError(f"Corresponding feature is missing shape metadata: {ft}")
        if isinstance(shape, list):
            shape = tuple(shape)
        elif not isinstance(shape, tuple):
            raise ValueError(f"Corresponding feature is not valid: {ft}")

        if dtype == "video":
            continue
        if dtype == "image":
            hf_features[key] = datasets.Image()
        elif len(shape) == 0:
            hf_features[key] = datasets.Value(dtype=dtype)
        elif shape == (1,):
            hf_features[key] = datasets.Value(dtype=dtype)
        elif len(shape) == 1:
            hf_features[key] = datasets.Sequence(
                length=shape[0], feature=datasets.Value(dtype=dtype)
            )
        elif len(shape) == 2:
            hf_features[key] = datasets.Array2D(shape=shape, dtype=dtype)
        elif len(shape) == 3:
            hf_features[key] = datasets.Array3D(shape=shape, dtype=dtype)
        elif len(shape) == 4:
            hf_features[key] = datasets.Array4D(shape=shape, dtype=dtype)
        elif len(shape) == 5:
            hf_features[key] = datasets.Array5D(shape=shape, dtype=dtype)
        else:
            raise ValueError(f"Corresponding feature is not valid: {ft}")

    return datasets.Features(hf_features)


class _MolmoLeRobotDataset(LeRobotDataset):
    def load_hf_dataset(self):
        # LeRobot datasets may declare metadata-only modalities like audio that are
        # not materialized in the parquet frame table and are unused by MolmoAct2.
        features = _get_hf_features_from_features_compatible(
            _filter_hf_loadable_features(self.features, repo_id=self.repo_id)
        )
        hf_dataset = load_nested_dataset(self.root / "data", features=features, episodes=self.episodes)
        hf_dataset.set_transform(hf_transform_to_torch)
        expected_num_frames = None
        meta = getattr(self, "meta", None)
        if meta is not None:
            if self.episodes is None:
                expected_num_frames = getattr(meta, "total_frames", None)
            else:
                try:
                    expected_num_frames = sum(
                        int(meta.episodes[int(ep_idx)]["dataset_to_index"])
                        - int(meta.episodes[int(ep_idx)]["dataset_from_index"])
                        for ep_idx in self.episodes
                    )
                except Exception:
                    expected_num_frames = None
        if (
            expected_num_frames is not None
            and int(expected_num_frames) != len(hf_dataset)
        ):
            _warn_frame_count_mismatch(
                self.repo_id,
                self.root / "data",
                len(hf_dataset),
                int(expected_num_frames),
            )
        return hf_dataset

    @property
    def num_frames(self) -> int:
        hf_dataset = getattr(self, "hf_dataset", None)
        if hf_dataset is not None:
            return len(hf_dataset)
        return super().num_frames


def _parse_repo_spec(raw_spec: str) -> _RepoSpec:
    spec = raw_spec.strip()
    if not spec:
        raise ValueError("Empty repo spec")
    if "@" not in spec:
        return _RepoSpec(repo_id=spec)
    repo_id, raw_episodes = spec.split("@", 1)
    episodes = _parse_episode_list(raw_episodes)
    return _RepoSpec(repo_id=repo_id.strip(), episodes=episodes)


def _parse_episode_list(raw_episodes: str) -> List[int]:
    episodes: List[int] = []
    for part in raw_episodes.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start_idx = int(start)
            end_idx = int(end)
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx
            episodes.extend(range(start_idx, end_idx + 1))
        else:
            episodes.append(int(part))
    if not episodes:
        raise ValueError(f"No episodes could be parsed from specification '{raw_episodes}'")
    return sorted(set(episodes))


def _normalize_episode_list(values: Iterable[Number]) -> List[int]:
    return sorted({int(v) for v in values})


def _get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("Invalid value '%s' for %s, using default %d", raw, name, default)
        return default
    return max(1, value)


def _get_env_non_negative_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("Invalid value '%s' for %s, using default %d", raw, name, default)
        return default
    return max(0, value)


def _get_env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _normalize_random_camera_order(value: Optional[str], *, label: str = "random_camera_order") -> str:
    mode = str(value or "none").strip().lower()
    if mode not in SUPPORTED_RANDOM_CAMERA_ORDER_MODES:
        raise ValueError(
            f"Unsupported {label}='{mode}'. Expected one of {sorted(SUPPORTED_RANDOM_CAMERA_ORDER_MODES)}."
        )
    return mode


def _require_positive_int(value: object, *, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a positive integer, got {value!r}.") from None
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive integer, got {parsed}.")
    return parsed


def _require_non_negative_int(value: object, *, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}.") from None
    if parsed < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {parsed}.")
    return parsed


def _get_required_env_int(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        raise ValueError(f"Missing required environment variable {name}.")
    return _require_positive_int(raw.strip(), label=name)


def _get_required_env_choice(name: str, *, choices: Sequence[str]) -> str:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        raise ValueError(
            f"Missing required environment variable {name}. Expected one of {sorted(choices)}."
        )
    value = raw.strip().lower()
    if value not in choices:
        raise ValueError(
            f"Unsupported {name}='{value}'. Expected one of {sorted(choices)}."
        )
    return value


def _resolve_required_positive_int(explicit_value: Optional[int], *, env_name: str, label: str) -> int:
    if explicit_value is None:
        return _get_required_env_int(env_name)
    return _require_positive_int(explicit_value, label=label)


def _require_non_empty_string_sequence(values: Sequence[str], *, label: str) -> List[str]:
    if not isinstance(values, Sequence) or not values:
        raise ValueError(f"{label} must be a non-empty sequence of strings.")
    normalized: List[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must contain only non-empty strings; got {value!r}.")
        normalized.append(value.strip())
    return normalized


def _normalize_optional_string_sequence(values: Optional[Sequence[str]], *, label: str) -> List[str]:
    if values is None:
        return []
    if not isinstance(values, Sequence):
        raise ValueError(f"{label} must be a sequence of strings if provided.")
    normalized: List[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must contain only non-empty strings; got {value!r}.")
        normalized.append(value.strip())
    return normalized


def _validate_available_camera_keys(
    available_cameras: Sequence[str],
    camera_keys: Sequence[str],
    *,
    repo_id: str,
    label: str,
) -> None:
    if not available_cameras or not camera_keys:
        return
    missing_camera_keys = [key for key in camera_keys if key not in available_cameras]
    if missing_camera_keys:
        raise ValueError(
            f"Camera keys {missing_camera_keys} required by {label} are missing for repo "
            f"'{repo_id}'. Available camera keys: {list(available_cameras)}"
        )


_YAM_DUAL_STANDARD_CAMERA_KEYS = [
    "observation.images.top",
    "observation.images.left",
    "observation.images.right",
]
_YAM_DUAL_LEGACY_CAMERA_KEYS = [
    "observation.images.camera_front",
    "observation.images.camera_left",
    "observation.images.camera_right",
]


def _resolve_repo_camera_keys(
    tag: str,
    *,
    available_cameras: Sequence[str],
    camera_keys: Sequence[str],
) -> List[str]:
    resolved_camera_keys = list(camera_keys)
    if str(tag).split(":", 1)[-1] != "yam_dual_molmoact2":
        return resolved_camera_keys
    if resolved_camera_keys != _YAM_DUAL_STANDARD_CAMERA_KEYS:
        return resolved_camera_keys
    available = set(str(key) for key in available_cameras)
    if set(_YAM_DUAL_STANDARD_CAMERA_KEYS).issubset(available):
        return resolved_camera_keys
    if set(_YAM_DUAL_LEGACY_CAMERA_KEYS).issubset(available):
        return list(_YAM_DUAL_LEGACY_CAMERA_KEYS)
    return resolved_camera_keys


def _require_non_empty_int_sequence(values: Sequence[int], *, label: str) -> List[int]:
    if not isinstance(values, Sequence) or not values:
        raise ValueError(f"{label} must be a non-empty sequence of integers.")
    normalized: List[int] = []
    for value in values:
        normalized.append(_require_non_negative_int(value, label=label))
    return normalized


def _get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_action_discrete_processor_from_local_snapshot(
    tokenizer_name_or_path: str,
    *,
    hf_token: Optional[str] = None,
):
    snapshot_path = Path(
        snapshot_download(
            tokenizer_name_or_path,
            local_files_only=True,
            token=hf_token,
        )
    )
    processor_config_path = snapshot_path / "processor_config.json"
    if not processor_config_path.is_file():
        raise FileNotFoundError(
            f"Cached processor config not found for discrete action tokenizer {tokenizer_name_or_path!r} "
            f"at {processor_config_path}."
        )

    processor_config = json.loads(processor_config_path.read_text(encoding="utf-8"))
    auto_processor_ref = (processor_config.get("auto_map") or {}).get("AutoProcessor")
    if not isinstance(auto_processor_ref, str) or "." not in auto_processor_ref:
        raise ValueError(
            f"Cached processor config for discrete action tokenizer {tokenizer_name_or_path!r} "
            f"does not define a loadable AutoProcessor entry."
        )

    module_name, class_name = auto_processor_ref.rsplit(".", 1)
    module_path = snapshot_path / f"{module_name}.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Cached processor module not found for discrete action tokenizer {tokenizer_name_or_path!r} "
            f"at {module_path}."
        )

    spec = importlib.util.spec_from_file_location(
        f"cached_action_processor_{hashlib.sha1(str(module_path).encode('utf-8')).hexdigest()}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import cached processor module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processor_cls = getattr(module, class_name)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
        token=hf_token,
    )
    processor_kwargs = {
        key: value
        for key, value in processor_config.items()
        if key not in {"auto_map", "processor_class"}
    }
    return processor_cls(tokenizer, **processor_kwargs)


def _parse_image_resize(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    for sep in ("x", ","):
        if sep in cleaned:
            parts = [p.strip() for p in cleaned.split(sep) if p.strip()]
            if len(parts) != 2:
                return None
            try:
                h = int(parts[0])
                w = int(parts[1])
            except ValueError:
                return None
            return (h, w) if h > 0 and w > 0 else None
    try:
        size = int(cleaned)
    except ValueError:
        return None
    return (size, size) if size > 0 else None


def _get_dist_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    for env_key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        raw = os.environ.get(env_key)
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                break
    return 0


def _wait_for_marker(marker_path: Path, timeout_s: int) -> None:
    start = time.time()
    while True:
        if marker_path.exists():
            return
        elapsed = time.time() - start
        if elapsed > timeout_s:
            raise RuntimeError(
                f"Timed out waiting for LeRobot download marker at {marker_path}"
            )
        time.sleep(2.0)


def _build_dataset_with_lock(
    dataset_root: Optional[Path],
    builder: Callable[[], LeRobotDataset],
) -> LeRobotDataset:
    if dataset_root is None:
        return builder()
    dataset_root.mkdir(parents=True, exist_ok=True)
    marker_path = dataset_root / ".lerobot_download_complete"
    lock_path = dataset_root / ".lerobot_download_lock"
    if marker_path.exists():
        return builder()

    rank = _get_dist_rank()
    timeout_s = _get_env_int("LEROBOT_DOWNLOAD_TIMEOUT_S", 1800)

    if rank == 0:
        acquired = False
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired = True
        except FileExistsError:
            acquired = False
        if not acquired:
            _wait_for_marker(marker_path, timeout_s)
            return builder()
        try:
            dataset = builder()
            marker_path.write_text("ok", encoding="utf-8")
            return dataset
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
    else:
        _wait_for_marker(marker_path, timeout_s)
        return builder()


def _get_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warning("Invalid value '%s' for %s, using default %s", raw, name, default)
        return default
    return float(value)


def _get_env_json(name: str) -> Optional[object]:
    raw = os.environ.get(name)
    if raw is not None:
        raw = raw.strip()
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                log.warning("Invalid JSON for %s, ignoring.", name)

    path = os.environ.get(f"{name}_PATH")
    if path is None:
        return None
    path = path.strip()
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        log.warning("JSON file for %s does not exist: %s", name, path)
    except json.JSONDecodeError:
        log.warning("Invalid JSON file for %s at %s, ignoring.", name, path)
    except OSError as exc:
        log.warning("Failed reading JSON file for %s at %s: %s", name, path, exc)
    return None


def _require_tag_metadata_entry(
    tag_metadata_by_tag: object,
    repo_to_tag: object,
    repo_id: str,
    *,
    random_camera_order: str = "none",
) -> Dict[str, Any]:
    random_camera_order = _normalize_random_camera_order(
        random_camera_order,
        label="random_camera_order",
    )
    if not isinstance(tag_metadata_by_tag, dict) or not tag_metadata_by_tag:
        raise ValueError(
            "LeRobot tag metadata is required. Set LEROBOT_TAG_METADATA with per-tag metadata."
        )
    if not isinstance(repo_to_tag, dict) or repo_id not in repo_to_tag:
        raise ValueError(
            f"Missing repo-to-tag mapping for LeRobot repo '{repo_id}'. "
            "Set LEROBOT_REPO_TO_TAG consistently with LEROBOT_TAG_METADATA."
        )
    tag = str(repo_to_tag[repo_id])
    metadata = tag_metadata_by_tag.get(tag)
    if not isinstance(metadata, dict):
        raise ValueError(f"Missing LeRobot tag metadata for tag '{tag}'.")

    required = [
        "action_key",
        "state_keys",
        "normalize_gripper",
        "action_horizon",
        "n_action_steps",
        "setup_type",
        "control_mode",
    ]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(
            f"LeRobot tag metadata for tag '{tag}' is missing required fields: {missing}"
        )
    camera_keys = metadata.get("camera_keys")
    if camera_keys is None:
        if random_camera_order == "none":
            raise ValueError(
                f"LeRobot tag metadata for tag '{tag}' must define a non-empty camera_keys list "
                "when random_camera_order='none'."
            )
    elif not isinstance(camera_keys, list) or not all(isinstance(v, str) and v for v in camera_keys):
        raise ValueError(f"LeRobot tag metadata for tag '{tag}' must define camera_keys as a list of non-empty strings.")
    elif not camera_keys and random_camera_order == "none":
        raise ValueError(
            f"LeRobot tag metadata for tag '{tag}' must define a non-empty camera_keys list "
            "when random_camera_order='none'."
        )
    camera_keys_alternative = metadata.get("camera_keys_alternative")
    if camera_keys_alternative is not None:
        if (
            not isinstance(camera_keys_alternative, list)
            or not all(isinstance(v, str) and v for v in camera_keys_alternative)
        ):
            raise ValueError(
                f"LeRobot tag metadata for tag '{tag}' must define camera_keys_alternative "
                "as a list of non-empty strings."
            )
        if not camera_keys_alternative:
            raise ValueError(
                f"LeRobot tag metadata for tag '{tag}' must define a non-empty "
                "camera_keys_alternative list when it is provided."
            )
        if not camera_keys:
            raise ValueError(
                f"LeRobot tag metadata for tag '{tag}' must define a non-empty camera_keys list "
                "when camera_keys_alternative is provided."
            )
    if not isinstance(metadata.get("normalize_gripper"), bool):
        raise ValueError(f"LeRobot tag metadata for tag '{tag}' must define boolean normalize_gripper.")
    for key_name in ("action_horizon", "n_action_steps"):
        value = metadata.get(key_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) < 1:
            raise ValueError(
                f"LeRobot tag metadata for tag '{tag}' must define integer {key_name} >= 1."
            )
    state_keys = metadata.get("state_keys")
    if not isinstance(state_keys, list) or not all(isinstance(v, str) and v for v in state_keys):
        raise ValueError(
            f"LeRobot tag metadata for tag '{tag}' must define state_keys as a list of non-empty strings."
        )
    if not state_keys:
        raise ValueError(
            f"LeRobot tag metadata for tag '{tag}' must define a non-empty state_keys list."
        )
    for key_name in ("action_key", "setup_type", "control_mode"):
        value = metadata.get(key_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"LeRobot tag metadata for tag '{tag}' must define non-empty {key_name}.")
    if int(metadata["n_action_steps"]) > int(metadata["action_horizon"]):
        raise ValueError(
            f"LeRobot tag metadata for tag '{tag}' has n_action_steps={int(metadata['n_action_steps'])}, "
            f"which exceeds action_horizon={int(metadata['action_horizon'])}."
        )
    return metadata


def _canonicalize_depth_repo_id(repo_id: str) -> str:
    repo_id = str(repo_id or "")
    if repo_id.endswith(_DEPTH_DATASET_SUFFIX):
        return repo_id[: -len(_DEPTH_DATASET_SUFFIX)]
    return repo_id


def _resolve_depth_dataset_root(base: str, dataset_repo_id: str) -> Path:
    base_path = Path(base).expanduser()
    primary = base_path / dataset_repo_id
    if primary.exists() or dataset_repo_id.endswith(_DEPTH_DATASET_SUFFIX):
        return primary
    suffixed = base_path / f"{dataset_repo_id}{_DEPTH_DATASET_SUFFIX}"
    return suffixed if suffixed.exists() else primary


def _normalize_style_sampling_rates(raw_rates: Optional[object]) -> Optional[Dict[str, float]]:
    if raw_rates is None:
        return None

    if isinstance(raw_rates, list):
        parsed: Dict[str, float] = {}
        for entry in raw_rates:
            if isinstance(entry, dict):
                parsed.update(entry)
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                parsed[str(entry[0])] = float(entry[1])
            else:
                raise ValueError(
                    "Style sampling rates list entries must be mappings or [style, rate] pairs."
                )
        raw_rates = parsed

    if not isinstance(raw_rates, dict):
        raise ValueError("Style sampling rates must be a dict or list of pairs.")

    cleaned: Dict[str, float] = {}
    for key, value in raw_rates.items():
        style = str(key).strip()
        if not style:
            continue
        if style not in ROBOT_OUTPUT_STYLES:
            raise ValueError(
                f"Unsupported LeRobot style '{style}' in style sampling rates. "
                f"Expected subset of {sorted(ROBOT_OUTPUT_STYLES)}."
            )
        rate = float(value)
        if rate <= 0:
            continue
        cleaned[style] = rate

    if not cleaned:
        return None

    total = sum(cleaned.values())
    if total <= 0:
        return None
    return {style: rate / total for style, rate in cleaned.items()}


def _should_skip_decode_error(exc: Exception) -> bool:
    if av is not None and isinstance(exc, av.error.InvalidDataError):
        return True
    message = str(exc).lower()
    return (
        "tolerance" in message
        or "query timestamps" in message
        or "invalid data found when processing input" in message
        or "avcodec_send_packet" in message
        or "no frame!" in message
    )


def _should_retry_with_main_revision(exc: Exception) -> bool:
    """Detect LeRobot version-resolution failures where forcing revision=main is safer."""
    message = str(exc)
    if isinstance(exc, NotImplementedError):
        return True
    return (
        "BackwardCompatibilityError" in message
        or "Contact the maintainer on [Discord]" in message
        or "dataset must be tagged with a codebase version" in message
    )


def _is_visual_observation(key: str) -> bool:
    return key.startswith("observation.images") or key.startswith("observation.image")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    elif torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value)
    else:
        array = np.asarray(value)
    return np.array(array)


def _image_to_uint8(array: np.ndarray) -> np.ndarray:
    arr = array
    if arr.ndim == 4:
        # delta timestamps can add an extra dimension
        arr = arr[-1]
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image tensor shape {array.shape}")
    # Convert to channel-last layout if needed
    if arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _resize_image_array(array: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = size
    arr = _image_to_uint8(array)
    if arr.shape[0] == target_h and arr.shape[1] == target_w:
        return arr
    if arr.shape[-1] == 1:
        pil_image = Image.fromarray(arr.squeeze(-1))
        resized = pil_image.resize((target_w, target_h), Image.BILINEAR)
        out = np.asarray(resized)
        if out.ndim == 2:
            out = out[:, :, None]
        return out
    pil_image = Image.fromarray(arr)
    resized = pil_image.resize((target_w, target_h), Image.BILINEAR)
    return np.asarray(resized)


def _resize_images(
    images: Union[np.ndarray, List[np.ndarray]],
    size: Tuple[int, int],
) -> Union[np.ndarray, List[np.ndarray]]:
    if isinstance(images, list):
        return [_resize_image_array(image, size) for image in images]
    return _resize_image_array(images, size)


def _prepare_array(value: Any, flatten: bool = False) -> np.ndarray:
    arr = _to_numpy(value).astype(np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if flatten:
        return arr.reshape(-1)
    return arr


def _collect_image_frames(array: Any) -> List[np.ndarray]:
    arr = _to_numpy(array)
    frames: List[np.ndarray] = []
    if arr.ndim == 5:
        # (time, camera, C, H, W) or similar
        merged = arr.reshape(-1, *arr.shape[-3:])
        for frame in merged:
            frames.append(_image_to_uint8(frame))
    elif arr.ndim == 4:
        # Either a stack of frames (time, C, H, W) or a single frame (C, H, W) with extra dim
        if arr.shape[1] in {1, 3, 4}:
            for frame in arr:
                frames.append(_image_to_uint8(frame))
        elif arr.shape[0] in {1, 3, 4}:
            frames.append(_image_to_uint8(arr))
        else:
            for frame in arr:
                frames.append(_image_to_uint8(frame))
    elif arr.ndim == 3:
        frames.append(_image_to_uint8(arr))
    else:
        raise ValueError(f"Unsupported image tensor shape {arr.shape}")
    return frames


def _resolve_delta_settings(
    repo_id: str,
    dataset_root: Optional[Path],
    n_obs_steps: int,
    action_horizon: int,
) -> Tuple[Optional[Dict[str, List[float]]], List[int], List[int]]:
    obs_indices = list(range(1 - n_obs_steps, 1))
    action_indices = list(range(1 - n_obs_steps, 1 - n_obs_steps + action_horizon))
    require_history = n_obs_steps > 1 or action_horizon > 1
    delta_timestamps: Optional[Dict[str, List[float]]] = None

    if require_history:
        metadata_kwargs = {
            "repo_id": repo_id,
            "root": str(dataset_root) if dataset_root else None,
        }
        try:
            meta = LeRobotDatasetMetadata(**metadata_kwargs)
        except Exception as exc:
            if not _should_retry_with_main_revision(exc):
                raise
            log.warning(
                "LeRobot metadata init failed for %s (%s). Retrying with revision='main'.",
                repo_id,
                exc,
            )
            meta = LeRobotDatasetMetadata(**metadata_kwargs, revision="main")
        fps = meta.fps
        timestamps: Dict[str, List[float]] = {}
        for key in meta.features:
            if n_obs_steps > 1 and key.startswith("observation."):
                timestamps[key] = [idx / fps for idx in obs_indices]
            if action_horizon > 1 and (key == "action" or key.startswith("action.")):
                timestamps[key] = [idx / fps for idx in action_indices]
        if timestamps:
            delta_timestamps = timestamps

    return delta_timestamps, obs_indices, action_indices


def _extract_vector(
    item: Dict[str, Any],
    preferred_keys: Sequence[str],
    prefix: Optional[str],
) -> Tuple[Optional[np.ndarray], Optional[Sequence[str]]]:
    explicit_keys = [str(key) for key in preferred_keys if key]
    values: List[np.ndarray] = []
    used_keys: List[str] = []
    missing_keys: List[str] = []
    for key in explicit_keys:
        if key in item:
            values.append(_prepare_array(item[key], flatten=len(explicit_keys) > 1))
            used_keys.append(key)
        else:
            missing_keys.append(key)
    if values:
        if missing_keys:
            raise ValueError(
                f"Missing configured vector keys {missing_keys}; found {used_keys}."
            )
        if len(values) == 1:
            return values[0], used_keys
        return np.concatenate(values, axis=0), used_keys
    if len(explicit_keys) > 1:
        raise ValueError(f"Missing configured vector keys {missing_keys}.")
    if prefix:
        values = []
        used_keys = []
        for key in item:
            if key.startswith(prefix):
                values.append(_prepare_array(item[key], flatten=True))
                used_keys.append(key)
        if values:
            return np.concatenate(values, axis=0), used_keys
    return None, None


def _combine_pad_masks(keys: Optional[Sequence[str]], example: Dict[str, Any]) -> Optional[np.ndarray]:
    if not keys:
        return None
    mask: Optional[np.ndarray] = None
    for key in keys:
        pad_key = f"{key}_is_pad"
        if pad_key not in example:
            continue
        pad_arr = _to_numpy(example[pad_key]).astype(np.bool_)
        if mask is None:
            mask = pad_arr
        else:
            if mask.shape != pad_arr.shape:
                raise ValueError(f"Mismatched pad mask shapes for {key}: {mask.shape} vs {pad_arr.shape}")
            mask = np.logical_or(mask, pad_arr)
    return mask


def _pad_action_to_max_horizon(
    action: Optional[np.ndarray],
    *,
    tag_action_horizon: int,
    max_action_horizon: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if action is None:
        return None, None
    if max_action_horizon < tag_action_horizon:
        raise ValueError(
            f"max_action_horizon ({max_action_horizon}) cannot be smaller than "
            f"tag_action_horizon ({tag_action_horizon})."
        )
    if action.ndim == 1:
        action = action[None, :]
    if action.ndim < 2:
        raise ValueError(f"Expected action tensor with rank >= 2 after time expansion, got {action.shape}.")
    if int(action.shape[0]) != int(tag_action_horizon):
        raise ValueError(
            f"Expected action sequence with horizon {int(tag_action_horizon)}, got {int(action.shape[0])}."
        )
    action_horizon_is_pad = np.zeros((max_action_horizon,), dtype=np.bool_)
    if max_action_horizon == tag_action_horizon:
        return action, action_horizon_is_pad
    padded_shape = (max_action_horizon, *action.shape[1:])
    padded = np.zeros(padded_shape, dtype=action.dtype)
    padded[:tag_action_horizon] = action
    action_horizon_is_pad[tag_action_horizon:] = True
    return padded, action_horizon_is_pad


def _validate_depth_side_channel_length(
    values: Any,
    *,
    expected: int,
    label: str,
) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if int(arr.shape[0]) != int(expected):
        raise ValueError(
            f"{label} must have length {int(expected)}, got {int(arr.shape[0])}."
        )
    return arr


class LeRobotDatasetWrapper(Dataset):
    """Converts LeRobotDataset frames into Molmo-style examples."""

    def __init__(
        self,
        dataset: LeRobotDataset,
        split: str,
        camera_keys: Optional[Sequence[str]],
        camera_keys_alternative: Optional[Sequence[str]],
        question_key: Optional[str],
        state_keys: Sequence[str],
        action_keys: Sequence[str],
        observation_indices: Sequence[int],
        action_indices: Sequence[int],
        drop_n_last_frames: int = 0,
        style: str = "demo",
        style_sampling_rates: Optional[Dict[str, float]] = None,
        action_format: str = "continuous",
        state_format: str = "continuous",
        num_discrete_state_tokens: int = DEFAULT_NUM_STATE_TOKENS,
        enable_depth_reasoning: bool = False,
        add_depth_tokens: bool = False,
        num_depth_tokens: int = DEFAULT_NUM_DEPTH_TOKENS,
        num_depth_tokens_per_image: int = 100,
        discrete_action_tokenizer: Optional[str] = None,
        max_discrete_action_token_id: Optional[int] = None,
        data_formatter_add_setup_tokens: bool = False,
        data_formatter_add_control_tokens: bool = False,
        random_camera_order: str = "none",
        random_camera_order_seed: Optional[int] = None,
        use_annotated_task: Optional[bool] = None,
        sample_annotated_task: Optional[bool] = None,
        tag_action_horizon: int = 1,
        tag_n_action_steps: int = 1,
        max_action_horizon: Optional[int] = None,
        max_action_dim: Optional[int] = None,
        robot_processor: Optional[RobotProcessor] = None,
        robot_processor_config: Optional[RobotProcessorConfig] = None,
        dataset_reopen_kwargs: Optional[Dict[str, Any]] = None,
        metadata_repo_id: Optional[str] = None,
        depth_dataset: Optional[LeRobotDataset] = None,
        depth_dataset_reopen_kwargs: Optional[Dict[str, Any]] = None,
        depth_dataset_reopen_kwargs_by_camera: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self._depth_dataset = depth_dataset
        self._depth_dataset_reopen_kwargs = dict(depth_dataset_reopen_kwargs or {})
        self._depth_datasets_by_camera: Dict[str, Optional[LeRobotDataset]] = {
            str(camera_key): None
            for camera_key in (depth_dataset_reopen_kwargs_by_camera or {})
        }
        self._depth_dataset_reopen_kwargs_by_camera: Dict[str, Dict[str, Any]] = {
            str(camera_key): dict(reopen_kwargs)
            for camera_key, reopen_kwargs in (depth_dataset_reopen_kwargs_by_camera or {}).items()
        }
        self._dataset_reopen_kwargs = dict(dataset_reopen_kwargs or {})
        if self.dataset is not None:
            self._dataset_reopen_kwargs.setdefault("repo_id", getattr(self.dataset, "repo_id", None))
            dataset_root = getattr(self.dataset, "root", None)
            self._dataset_reopen_kwargs.setdefault(
                "root",
                None if dataset_root is None else str(dataset_root),
            )
            self._dataset_reopen_kwargs["revision"] = getattr(
                self.dataset,
                "revision",
                self._dataset_reopen_kwargs.get("revision"),
            )
        self.split = split
        self.random_camera_order = _normalize_random_camera_order(random_camera_order)
        self.camera_keys = _normalize_optional_string_sequence(camera_keys, label="camera_keys")
        self.camera_keys_alternative = _normalize_optional_string_sequence(
            camera_keys_alternative,
            label="camera_keys_alternative",
        )
        if self.camera_keys_alternative and not self.camera_keys:
            raise ValueError("camera_keys_alternative requires camera_keys to also be provided.")
        if self.random_camera_order == "none" and not self.camera_keys:
            raise ValueError("camera_keys must be provided when random_camera_order='none'.")
        self.random_camera_order_seed = _require_non_negative_int(
            0 if random_camera_order_seed is None else random_camera_order_seed,
            label="random_camera_order_seed",
        )
        meta = getattr(self.dataset, "meta", None)
        self._meta_camera_keys = _normalize_optional_string_sequence(
            getattr(meta, "camera_keys", None),
            label="dataset.meta.camera_keys",
        )
        self.question_key = question_key
        self.use_annotated_task = (
            _get_env_bool("LEROBOT_USE_ANNOTATED_TASK", False)
            if use_annotated_task is None
            else bool(use_annotated_task)
        )
        self.sample_annotated_task = (
            _get_env_bool("LEROBOT_SAMPLE_ANNOTATED_TASK", False)
            if sample_annotated_task is None
            else bool(sample_annotated_task)
        )
        if self.sample_annotated_task and not self.use_annotated_task:
            _warn_annotated_task(
                self._get_repo_id(),
                "sample_annotated_requires_use_annotated_task",
                "sample_annotated_task was enabled for repo '%s', but use_annotated_task is disabled. "
                "Ignoring sampled annotated-task mode.",
                self._get_repo_id(),
            )
            self.sample_annotated_task = False
        self.state_keys = _require_non_empty_string_sequence(state_keys, label="state_keys")
        self.action_keys = _require_non_empty_string_sequence(action_keys, label="action_keys")
        self.style = style
        self.style_sampling_rates = _normalize_style_sampling_rates(style_sampling_rates)
        self._sampled_styles: Optional[List[str]] = None
        self._sampled_style_probs: Optional[np.ndarray] = None
        if self.style_sampling_rates:
            self._sampled_styles = list(self.style_sampling_rates.keys())
            self._sampled_style_probs = np.asarray(
                [self.style_sampling_rates[s] for s in self._sampled_styles],
                dtype=np.float64,
            )
        self.action_format = action_format.strip().lower()
        if self.action_format not in SUPPORTED_ACTION_FORMATS:
            raise ValueError(
                f"Unsupported action_format='{self.action_format}'. "
                f"Expected one of {sorted(SUPPORTED_ACTION_FORMATS)}."
            )
        self.state_format = state_format.strip().lower()
        if self.state_format not in SUPPORTED_STATE_FORMATS:
            raise ValueError(
                f"Unsupported state_format='{self.state_format}'. "
                f"Expected one of {sorted(SUPPORTED_STATE_FORMATS)}."
            )
        self.num_discrete_state_tokens = int(num_discrete_state_tokens)
        if self.state_format in {"discrete", "both"} and self.num_discrete_state_tokens <= 0:
            raise ValueError(
                f"Discrete state_format requires num_discrete_state_tokens > 0, "
                f"got {self.num_discrete_state_tokens}."
            )
        self.enable_depth_reasoning = bool(enable_depth_reasoning)
        self.add_depth_tokens = bool(add_depth_tokens)
        self.num_depth_tokens = int(num_depth_tokens)
        self.num_depth_tokens_per_image = int(num_depth_tokens_per_image)
        if self.add_depth_tokens and self.num_depth_tokens <= 0:
            raise ValueError(
                f"add_depth_tokens requires num_depth_tokens > 0, got {self.num_depth_tokens}."
            )
        if self.num_depth_tokens_per_image <= 0:
            raise ValueError(
                f"num_depth_tokens_per_image must be > 0, got {self.num_depth_tokens_per_image}."
            )
        depth_styles_active = any(
            style_uses_depth_output(style_name)
            for style_name in (
                self._sampled_styles
                if self._sampled_styles is not None
                else [self.style]
            )
        )
        if depth_styles_active and not self.enable_depth_reasoning:
            raise ValueError("Depth output styles require enable_depth_reasoning=True.")
        if depth_styles_active and not self.add_depth_tokens:
            raise ValueError("Depth output styles require add_depth_tokens=True.")
        normalized_discrete_action_tokenizer = str(discrete_action_tokenizer or "").strip() or None
        if self.action_format in {"discrete", "both"} and normalized_discrete_action_tokenizer is None:
            raise ValueError(
                "LeRobotDatasetWrapper requires `discrete_action_tokenizer` when "
                "action_format is 'discrete' or 'both'."
            )
        self.discrete_action_tokenizer = normalized_discrete_action_tokenizer
        self.max_discrete_action_token_id = (
            None
            if max_discrete_action_token_id is None
            else int(max_discrete_action_token_id)
        )
        self.data_formatter_add_setup_tokens = bool(data_formatter_add_setup_tokens)
        self.data_formatter_add_control_tokens = bool(data_formatter_add_control_tokens)
        self.observation_delta_indices = _require_non_empty_int_sequence(
            observation_indices,
            label="observation_indices",
        )
        self.action_delta_indices = _require_non_empty_int_sequence(
            action_indices,
            label="action_indices",
        )
        self.drop_n_last_frames = _require_non_negative_int(
            drop_n_last_frames,
            label="drop_n_last_frames",
        )
        self.tag_action_horizon = _require_positive_int(
            tag_action_horizon,
            label="tag_action_horizon",
        )
        self.tag_n_action_steps = _require_positive_int(
            tag_n_action_steps,
            label="tag_n_action_steps",
        )
        if self.tag_n_action_steps > self.tag_action_horizon:
            raise ValueError(
                f"tag_n_action_steps ({self.tag_n_action_steps}) cannot exceed "
                f"tag_action_horizon ({self.tag_action_horizon})."
            )
        self.max_action_horizon = _require_positive_int(
            self.tag_action_horizon if max_action_horizon is None else max_action_horizon,
            label="max_action_horizon",
        )
        if self.max_action_horizon < self.tag_action_horizon:
            raise ValueError(
                f"max_action_horizon ({self.max_action_horizon}) cannot be smaller than "
                f"tag_action_horizon ({self.tag_action_horizon})."
            )
        self.max_action_dim = (
            None
            if max_action_dim is None
            else _require_positive_int(max_action_dim, label="max_action_dim")
        )
        if (
            self.action_format in {"discrete", "both"}
            and self.max_action_dim is not None
            and self.max_action_dim > ACTION_TOKENIZER_MAX_ACTION_DIM
        ):
            raise ValueError(
                "The action tokenizer is only trained with action dim max to 32; "
                "action_format='both' and action_format='discrete' are not supported "
                f"when max_action_dim={self.max_action_dim}. Use action_format='continuous'."
            )
        self.metadata_repo_id = _canonicalize_depth_repo_id(
            str(metadata_repo_id or getattr(self.dataset, "repo_id", "") or "")
        )
        self.robot_processor = robot_processor
        self.robot_processor_config = robot_processor_config
        self._warned_missing_state = False
        self._warned_missing_action = False
        self._warned_missing_question = False
        self._warned_discrete_oob = False
        self._warned_camera_order_episode_fallback = False
        self._episode_ranges: Optional[List[Tuple[int, int]]] = None
        self._episode_cumulative: Optional[List[int]] = None
        self._effective_len: Optional[int] = None
        self._cached_dataset_len: Optional[int] = len(self.dataset) if self.dataset is not None else None
        self._action_discrete_processor = None
        self._annotated_tasks_loaded = False
        self._annotated_tasks_by_episode: Optional[Dict[int, Any]] = None
        self._task_to_episode_loaded = False
        self._task_to_episode_by_task_index: Optional[Dict[int, np.ndarray]] = None
        self._build_episode_index()

    def __repr__(self) -> str:
        try:
            repo_id = self._get_repo_id()
        except Exception:
            repo_id = "unknown"
        return f"LeRobotDatasetWrapper(repo_id={repo_id!r}, split={self.split!r})"

    def _ensure_dataset_loaded(self) -> None:
        if self.dataset is not None:
            return
        dataset_kwargs = dict(self._dataset_reopen_kwargs)
        repo_id = dataset_kwargs.get("repo_id")
        if not repo_id:
            raise RuntimeError("Missing LeRobot dataset reopen configuration.")
        dataset_root = dataset_kwargs.get("root")
        dataset_root_path = Path(dataset_root) if dataset_root else None

        def _build_dataset() -> LeRobotDataset:
            try:
                return _MolmoLeRobotDataset(**dataset_kwargs)
            except Exception as exc:
                if not _should_retry_with_main_revision(exc):
                    raise
                log.warning(
                    "LeRobot dataset reopen failed for %s (%s). Retrying with revision='main'.",
                    repo_id,
                    exc,
                )
                return _MolmoLeRobotDataset(**dataset_kwargs, revision="main")

        self.dataset = _build_dataset_with_lock(dataset_root_path, _build_dataset)
        self._cached_dataset_len = len(self.dataset)
        self._dataset_reopen_kwargs["revision"] = getattr(
            self.dataset,
            "revision",
            self._dataset_reopen_kwargs.get("revision"),
        )
        self._episode_ranges = None
        self._episode_cumulative = None
        self._effective_len = None
        self._build_episode_index()

    def _load_depth_dataset_from_kwargs(self, depth_kwargs: Dict[str, Any]) -> LeRobotDataset:
        repo_id = depth_kwargs.get("repo_id")
        if not repo_id:
            raise RuntimeError("Missing LeRobot depth dataset reopen configuration.")

        import huggingface_hub.constants as _hf_consts

        previous_hf_offline = _hf_consts.HF_HUB_OFFLINE
        _hf_consts.HF_HUB_OFFLINE = True
        try:
            try:
                return _MolmoLeRobotDataset(**depth_kwargs)
            except Exception as exc:
                if not _should_retry_with_main_revision(exc):
                    raise
                return _MolmoLeRobotDataset(**depth_kwargs, revision="main")
        finally:
            _hf_consts.HF_HUB_OFFLINE = previous_hf_offline

    def _ensure_depth_dataset_loaded(self) -> None:
        if self._depth_dataset is not None or not self._depth_dataset_reopen_kwargs:
            return
        depth_kwargs = dict(self._depth_dataset_reopen_kwargs)
        self._depth_dataset = self._load_depth_dataset_from_kwargs(depth_kwargs)

    def _ensure_depth_dataset_for_camera_loaded(self, camera_key: str) -> Optional[LeRobotDataset]:
        camera_key = str(camera_key)
        if camera_key not in self._depth_dataset_reopen_kwargs_by_camera:
            return None
        depth_dataset = self._depth_datasets_by_camera.get(camera_key)
        if depth_dataset is not None:
            return depth_dataset
        depth_kwargs = dict(self._depth_dataset_reopen_kwargs_by_camera[camera_key])
        try:
            depth_dataset = self._load_depth_dataset_from_kwargs(depth_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load depth companion dataset for camera {camera_key!r} "
                f"from {depth_kwargs.get('root')!r}."
            ) from exc
        self._depth_datasets_by_camera[camera_key] = depth_dataset
        self._depth_dataset_reopen_kwargs_by_camera[camera_key] = dict(
            depth_kwargs,
            revision=getattr(depth_dataset, "revision", depth_kwargs.get("revision")),
        )
        return depth_dataset

    def _select_depth_dataset_for_cameras(
        self,
        camera_keys_used: Sequence[str],
    ) -> Tuple[Optional[str], Optional[LeRobotDataset]]:
        for camera_key in camera_keys_used:
            if camera_key in self._depth_dataset_reopen_kwargs_by_camera:
                return camera_key, self._ensure_depth_dataset_for_camera_loaded(camera_key)
        self._ensure_depth_dataset_loaded()
        return None, self._depth_dataset

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        dataset = state.get("dataset")
        if dataset is not None:
            reopen_kwargs = dict(state.get("_dataset_reopen_kwargs") or {})
            reopen_kwargs.setdefault("repo_id", getattr(dataset, "repo_id", None))
            dataset_root = getattr(dataset, "root", None)
            reopen_kwargs.setdefault("root", None if dataset_root is None else str(dataset_root))
            reopen_kwargs["revision"] = getattr(dataset, "revision", reopen_kwargs.get("revision"))
            state["_dataset_reopen_kwargs"] = reopen_kwargs
            try:
                state["_cached_dataset_len"] = len(dataset)
            except Exception:
                pass
            state["dataset"] = None
        depth_dataset = state.get("_depth_dataset")
        if depth_dataset is not None:
            reopen_kwargs = dict(state.get("_depth_dataset_reopen_kwargs") or {})
            reopen_kwargs.setdefault("repo_id", getattr(depth_dataset, "repo_id", None))
            depth_root = getattr(depth_dataset, "root", None)
            reopen_kwargs.setdefault("root", None if depth_root is None else str(depth_root))
            reopen_kwargs["revision"] = getattr(depth_dataset, "revision", reopen_kwargs.get("revision"))
            state["_depth_dataset_reopen_kwargs"] = reopen_kwargs
            state["_depth_dataset"] = None
        depth_datasets_by_camera = state.get("_depth_datasets_by_camera") or {}
        depth_reopen_by_camera = {
            str(camera_key): dict(reopen_kwargs)
            for camera_key, reopen_kwargs in (state.get("_depth_dataset_reopen_kwargs_by_camera") or {}).items()
        }
        for camera_key, depth_dataset in depth_datasets_by_camera.items():
            if depth_dataset is None:
                continue
            reopen_kwargs = dict(depth_reopen_by_camera.get(camera_key) or {})
            reopen_kwargs.setdefault("repo_id", getattr(depth_dataset, "repo_id", None))
            depth_root = getattr(depth_dataset, "root", None)
            reopen_kwargs.setdefault("root", None if depth_root is None else str(depth_root))
            reopen_kwargs["revision"] = getattr(depth_dataset, "revision", reopen_kwargs.get("revision"))
            depth_reopen_by_camera[str(camera_key)] = reopen_kwargs
        state["_depth_dataset_reopen_kwargs_by_camera"] = depth_reopen_by_camera
        state["_depth_datasets_by_camera"] = {
            camera_key: None for camera_key in depth_reopen_by_camera
        }
        state["_action_discrete_processor"] = None
        if not state.get("_annotated_tasks_loaded", False):
            state["_annotated_tasks_loaded"] = False
            state["_annotated_tasks_by_episode"] = None
        if not state.get("_task_to_episode_loaded", False):
            state["_task_to_episode_loaded"] = False
            state["_task_to_episode_by_task_index"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("_depth_datasets_by_camera", {})
        self.__dict__.setdefault("_depth_dataset_reopen_kwargs_by_camera", {})

    def _get_action_discrete_processor(self):
        if self._action_discrete_processor is None:
            if not self.discrete_action_tokenizer:
                raise ValueError(
                    "Discrete action processing requires `discrete_action_tokenizer` to be set."
                )
            offline_mode = (
                _get_env_bool("HF_HUB_OFFLINE", False)
                or _get_env_bool("TRANSFORMERS_OFFLINE", False)
                or _get_env_bool("HF_DATASETS_OFFLINE", False)
            )
            common_kwargs: Dict[str, Any] = {"trust_remote_code": True}
            hf_token = get_hf_access_token()
            cache_key = _action_discrete_processor_cache_key(
                self.discrete_action_tokenizer,
                offline_mode=offline_mode,
                hf_token=hf_token,
            )
            cached_processor = _ACTION_DISCRETE_PROCESSOR_CACHE.get(cache_key)
            if cached_processor is not None:
                self._action_discrete_processor = cached_processor
                return cached_processor
            if hf_token:
                common_kwargs["token"] = hf_token
            if offline_mode:
                try:
                    processor_path = Path(self.discrete_action_tokenizer).expanduser()
                    if processor_path.exists():
                        processor = AutoProcessor.from_pretrained(
                            str(processor_path),
                            local_files_only=True,
                            **common_kwargs,
                        )
                    else:
                        processor = _load_action_discrete_processor_from_local_snapshot(
                            self.discrete_action_tokenizer,
                            hf_token=hf_token,
                        )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load discrete action tokenizer '{self.discrete_action_tokenizer}' "
                        "in offline mode."
                    ) from exc
            else:
                try:
                    processor = AutoProcessor.from_pretrained(
                        self.discrete_action_tokenizer,
                        **common_kwargs,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load discrete action tokenizer '{self.discrete_action_tokenizer}'."
                    ) from exc
            self._action_discrete_processor = processor
            _ACTION_DISCRETE_PROCESSOR_CACHE[cache_key] = processor
        return self._action_discrete_processor

    def _warn_discrete_action_oob(self, token_ids: List[int]) -> None:
        if self._warned_discrete_oob:
            return
        log.warning(
            "Discrete action tokenizer produced token ids outside [0, %d): sample=%s. "
            "These tokens may be unknown to the current tokenizer vocabulary.",
            self.max_discrete_action_token_id,
            token_ids[:8],
        )
        self._warned_discrete_oob = True

    def _sample_style(self, rng: np.random.Generator) -> str:
        if self._sampled_styles is None or self._sampled_style_probs is None:
            return self.style
        sampled_index = int(rng.choice(len(self._sampled_styles), p=self._sampled_style_probs))
        return self._sampled_styles[sampled_index]

    def _compute_task_progress(self, frame: Dict[str, Any], mapped_index: Optional[int]) -> float:
        frame_index = frame.get("frame_index")
        episode_index = frame.get("episode_index")
        if torch.is_tensor(frame_index):
            frame_index = int(frame_index.item())
        elif frame_index is not None:
            frame_index = int(frame_index)
        if torch.is_tensor(episode_index):
            episode_index = int(episode_index.item())
        elif episode_index is not None:
            episode_index = int(episode_index)

        if frame_index is None or episode_index is None:
            if mapped_index is None:
                return 0.0
            meta = getattr(self.dataset, "meta", None)
            if meta is None or not hasattr(meta, "episodes"):
                return 0.0
            episode_ranges = getattr(meta, "episodes", None)
            if episode_ranges is None:
                return 0.0
            for ep in episode_ranges:
                start = int(ep["dataset_from_index"])
                end = int(ep["dataset_to_index"])
                if start <= mapped_index < end:
                    frame_index = mapped_index - start
                    total_frames = max(end - start, 1)
                    return float(frame_index) / float(max(total_frames - 1, 1))
            return 0.0

        meta = getattr(self.dataset, "meta", None)
        if meta is None or not hasattr(meta, "episodes"):
            return 0.0
        try:
            ep = meta.episodes[episode_index]
        except Exception:
            return 0.0
        if "length" in ep:
            total_frames = int(ep["length"])
        else:
            total_frames = int(ep["dataset_to_index"]) - int(ep["dataset_from_index"])
        return float(frame_index) / float(max(total_frames - 1, 1))

    def __len__(self) -> int:
        self._ensure_dataset_loaded()
        if self._effective_len is not None:
            return self._effective_len
        return len(self.dataset)

    def _get_repo_id(self) -> str:
        if self.metadata_repo_id:
            return self.metadata_repo_id
        if self.dataset is None:
            repo_id = self._dataset_reopen_kwargs.get("repo_id")
            if isinstance(repo_id, str) and repo_id:
                return repo_id
            raise RuntimeError("LeRobot dataset repo_id is unavailable.")
        return self.dataset.repo_id

    def _resolve_episode_index(
        self,
        frame: Dict[str, Any],
        mapped_index: Optional[int],
    ) -> Optional[int]:
        episode_index = frame.get("episode_index")
        if torch.is_tensor(episode_index):
            episode_index = int(episode_index.item())
        elif episode_index is not None:
            episode_index = int(episode_index)
        if episode_index is not None:
            return episode_index
        if mapped_index is None:
            return None
        meta = getattr(self.dataset, "meta", None)
        episode_ranges = getattr(meta, "episodes", None) if meta is not None else None
        if episode_ranges is None:
            return None
        for idx, ep in enumerate(episode_ranges):
            start = int(ep["dataset_from_index"])
            end = int(ep["dataset_to_index"])
            if start <= mapped_index < end:
                return idx
        return None

    def _get_dataset_root_path(self) -> Optional[Path]:
        dataset_root = None
        if self.dataset is not None:
            dataset_root = getattr(self.dataset, "root", None)
        if dataset_root is None:
            dataset_root = self._dataset_reopen_kwargs.get("root")
        if dataset_root is None:
            return None
        return Path(dataset_root)

    def _set_cached_annotated_tasks(
        self,
        cache_key: Tuple[str, str],
        tasks_by_episode: Optional[Dict[int, Any]],
    ) -> Optional[Dict[int, Any]]:
        self._annotated_tasks_by_episode = tasks_by_episode
        _ANNOTATED_TASKS_CACHE[cache_key] = tasks_by_episode
        return self._annotated_tasks_by_episode

    def _set_cached_task_to_episode_pool(
        self,
        cache_key: Tuple[str, str],
        pool_by_task_index: Optional[Dict[int, np.ndarray]],
    ) -> Optional[Dict[int, np.ndarray]]:
        self._task_to_episode_by_task_index = pool_by_task_index
        _TASK_TO_EPISODE_CACHE[cache_key] = pool_by_task_index
        return self._task_to_episode_by_task_index

    def _load_annotated_tasks(self) -> Optional[Dict[int, Any]]:
        if not self.use_annotated_task:
            return None
        if self._annotated_tasks_loaded:
            return self._annotated_tasks_by_episode

        self._annotated_tasks_loaded = True
        repo_id = self._get_repo_id()
        dataset_root = self._get_dataset_root_path()
        if dataset_root is None:
            _warn_annotated_task(
                repo_id,
                "missing_dataset_root",
                "Annotated-task loading enabled for repo '%s', but the dataset root is unavailable. "
                "Falling back to original LeRobot tasks.",
                repo_id,
            )
            return None

        annotated_path = dataset_root / "meta" / _ANNOTATED_TASKS_FILENAME
        cache_key = _annotated_tasks_cache_key(repo_id, annotated_path)
        if cache_key in _ANNOTATED_TASKS_CACHE:
            self._annotated_tasks_by_episode = _ANNOTATED_TASKS_CACHE[cache_key]
            return self._annotated_tasks_by_episode

        if not annotated_path.is_file():
            _warn_annotated_task(
                repo_id,
                "missing_annotated_file",
                "Annotated-task loading enabled for repo '%s', but %s is missing. "
                "Falling back to original LeRobot tasks.",
                repo_id,
                annotated_path,
            )
            return self._set_cached_annotated_tasks(cache_key, None)

        try:
            annotated_df = pd.read_parquet(annotated_path)
        except Exception as exc:
            _warn_annotated_task(
                repo_id,
                "annotated_file_read_failed",
                "Failed to read annotated tasks for repo '%s' from %s (%s). "
                "Falling back to original LeRobot tasks.",
                repo_id,
                annotated_path,
                exc,
            )
            return self._set_cached_annotated_tasks(cache_key, None)

        if annotated_df.index.name != _ANNOTATED_TASKS_INDEX_NAME:
            _warn_annotated_task(
                repo_id,
                "annotated_file_bad_index",
                "Annotated tasks for repo '%s' at %s use index %r instead of %r. "
                "Falling back to original LeRobot tasks.",
                repo_id,
                annotated_path,
                annotated_df.index.name,
                _ANNOTATED_TASKS_INDEX_NAME,
            )
            return self._set_cached_annotated_tasks(cache_key, None)

        if list(annotated_df.columns) != [_ANNOTATED_TASKS_COLUMN_NAME]:
            _warn_annotated_task(
                repo_id,
                "annotated_file_bad_columns",
                "Annotated tasks for repo '%s' at %s use columns %s instead of [%r]. "
                "Falling back to original LeRobot tasks.",
                repo_id,
                annotated_path,
                list(annotated_df.columns),
                _ANNOTATED_TASKS_COLUMN_NAME,
            )
            return self._set_cached_annotated_tasks(cache_key, None)

        if annotated_df.index.has_duplicates:
            _warn_annotated_task(
                repo_id,
                "annotated_file_duplicate_episode_index",
                "Annotated tasks for repo '%s' at %s contain duplicate episode indices. "
                "Falling back to original LeRobot tasks.",
                repo_id,
                annotated_path,
            )
            return self._set_cached_annotated_tasks(cache_key, None)

        tasks_by_episode: Dict[int, Any] = {}
        for raw_episode_index, task_value in annotated_df[_ANNOTATED_TASKS_COLUMN_NAME].items():
            try:
                episode_index = int(raw_episode_index)
            except (TypeError, ValueError):
                _warn_annotated_task(
                    repo_id,
                    "annotated_file_non_integer_episode_index",
                    "Annotated tasks for repo '%s' at %s contain a non-integer episode index %r. "
                    "Falling back to original LeRobot tasks.",
                    repo_id,
                    annotated_path,
                    raw_episode_index,
                )
                return self._set_cached_annotated_tasks(cache_key, None)
            tasks_by_episode[episode_index] = task_value

        return self._set_cached_annotated_tasks(cache_key, tasks_by_episode)

    def preload_annotated_tasks(self) -> Optional[Dict[int, Any]]:
        return self._load_annotated_tasks()

    def _load_task_to_episode_pool(self) -> Optional[Dict[int, np.ndarray]]:
        if not self.use_annotated_task or not self.sample_annotated_task:
            return None
        if self._task_to_episode_loaded:
            return self._task_to_episode_by_task_index

        self._task_to_episode_loaded = True
        repo_id = self._get_repo_id()
        dataset_root = self._get_dataset_root_path()
        if dataset_root is None:
            _warn_annotated_task(
                repo_id,
                "missing_dataset_root_for_task_to_episode",
                "Sampled annotated-task loading enabled for repo '%s', but the dataset root is unavailable. "
                "Falling back to per-episode annotated tasks.",
                repo_id,
            )
            return None

        pool_path = dataset_root / "meta" / _TASK_TO_EPISODE_FILENAME
        cache_key = _annotated_tasks_cache_key(repo_id, pool_path)
        if cache_key in _TASK_TO_EPISODE_CACHE:
            self._task_to_episode_by_task_index = _TASK_TO_EPISODE_CACHE[cache_key]
            return self._task_to_episode_by_task_index

        if not pool_path.is_file():
            _warn_annotated_task(
                repo_id,
                "missing_task_to_episode_file",
                "Sampled annotated-task loading enabled for repo '%s', but %s is missing. "
                "Falling back to per-episode annotated tasks.",
                repo_id,
                pool_path,
            )
            return self._set_cached_task_to_episode_pool(cache_key, None)

        try:
            pool_df = pd.read_parquet(pool_path)
        except Exception as exc:
            _warn_annotated_task(
                repo_id,
                "task_to_episode_read_failed",
                "Failed to read sampled annotated-task pool for repo '%s' from %s (%s). "
                "Falling back to per-episode annotated tasks.",
                repo_id,
                pool_path,
                exc,
            )
            return self._set_cached_task_to_episode_pool(cache_key, None)

        if list(pool_df.columns) != [
            _TASK_TO_EPISODE_TASK_INDEX_COLUMN,
            _TASK_TO_EPISODE_EPISODE_INDEX_COLUMN,
        ]:
            _warn_annotated_task(
                repo_id,
                "task_to_episode_bad_columns",
                "Sampled annotated-task pool for repo '%s' at %s uses columns %s instead of [%r, %r]. "
                "Falling back to per-episode annotated tasks.",
                repo_id,
                pool_path,
                list(pool_df.columns),
                _TASK_TO_EPISODE_TASK_INDEX_COLUMN,
                _TASK_TO_EPISODE_EPISODE_INDEX_COLUMN,
            )
            return self._set_cached_task_to_episode_pool(cache_key, None)

        pairs_by_task_index: Dict[int, List[int]] = {}
        try:
            deduped_df = pool_df.drop_duplicates(ignore_index=True)
            for task_index_value, episode_index_value in deduped_df.itertuples(index=False, name=None):
                task_index = int(task_index_value)
                episode_index = int(episode_index_value)
                pairs_by_task_index.setdefault(task_index, []).append(episode_index)
        except Exception as exc:
            _warn_annotated_task(
                repo_id,
                "task_to_episode_bad_values",
                "Sampled annotated-task pool for repo '%s' at %s contains non-integer task/episode ids (%s). "
                "Falling back to per-episode annotated tasks.",
                repo_id,
                pool_path,
                exc,
            )
            return self._set_cached_task_to_episode_pool(cache_key, None)

        if not pairs_by_task_index:
            _warn_annotated_task(
                repo_id,
                "task_to_episode_empty",
                "Sampled annotated-task pool for repo '%s' at %s is empty. "
                "Falling back to per-episode annotated tasks.",
                repo_id,
                pool_path,
            )
            return self._set_cached_task_to_episode_pool(cache_key, None)

        pool_by_task_index = {
            task_index: np.asarray(sorted(set(episode_indices)), dtype=np.int64)
            for task_index, episode_indices in pairs_by_task_index.items()
        }
        return self._set_cached_task_to_episode_pool(cache_key, pool_by_task_index)

    def preload_task_to_episode_pool(self) -> Optional[Dict[int, np.ndarray]]:
        return self._load_task_to_episode_pool()

    def _resolve_task_index(self, frame: Dict[str, Any]) -> Optional[int]:
        task_index = frame.get("task_index")
        if torch.is_tensor(task_index):
            task_index = int(task_index.item())
        elif task_index is not None:
            task_index = int(task_index)
        if task_index is None:
            return None
        return int(task_index)

    def _resolve_annotated_task_for_episode_index(
        self,
        episode_index: int,
        *,
        missing_row_warning_kind: str,
        missing_row_message: str,
        invalid_type_warning_kind: str,
        invalid_type_message: str,
        blank_warning_kind: str,
        blank_message: str,
    ) -> Tuple[Optional[str], bool]:
        tasks_by_episode = self._load_annotated_tasks()
        if tasks_by_episode is None:
            return None, False

        repo_id = self._get_repo_id()
        if episode_index not in tasks_by_episode:
            _warn_annotated_task(
                repo_id,
                missing_row_warning_kind,
                missing_row_message,
                repo_id,
                episode_index,
                _ANNOTATED_TASKS_FILENAME,
            )
            return None, False

        task_value = tasks_by_episode[episode_index]
        if task_value == _FROZEN_TASK_SENTINEL:
            return None, True
        if not isinstance(task_value, str):
            _warn_annotated_task(
                repo_id,
                invalid_type_warning_kind,
                invalid_type_message,
                repo_id,
                episode_index,
                task_value,
            )
            return None, False
        if not task_value.strip():
            _warn_annotated_task(
                repo_id,
                blank_warning_kind,
                blank_message,
                repo_id,
                episode_index,
            )
            return None, False
        return task_value, False

    def _resolve_annotated_task(
        self,
        frame: Dict[str, Any],
        mapped_index: Optional[int],
    ) -> Tuple[Optional[str], bool]:
        tasks_by_episode = self._load_annotated_tasks()
        if tasks_by_episode is None:
            return None, False

        repo_id = self._get_repo_id()
        episode_index = self._resolve_episode_index(frame, mapped_index)
        if episode_index is None:
            _warn_annotated_task(
                repo_id,
                "annotated_missing_episode_index",
                "Annotated-task loading enabled for repo '%s', but episode_index could not be resolved. "
                "Falling back to original LeRobot tasks.",
                repo_id,
            )
            return None, False
        return self._resolve_annotated_task_for_episode_index(
            episode_index,
            missing_row_warning_kind="annotated_missing_episode_row",
            missing_row_message=(
                "Annotated-task loading enabled for repo '%s', but episode_index=%d is missing from %s. "
                "Falling back to original LeRobot tasks."
            ),
            invalid_type_warning_kind="annotated_non_string_task",
            invalid_type_message=(
                "Annotated-task loading enabled for repo '%s', but episode_index=%d has a non-string task %r. "
                "Falling back to original LeRobot tasks."
            ),
            blank_warning_kind="annotated_blank_task",
            blank_message=(
                "Annotated-task loading enabled for repo '%s', but episode_index=%d has an empty task. "
                "Falling back to original LeRobot tasks."
            ),
        )

    def _sample_annotated_task(
        self,
        frame: Dict[str, Any],
        mapped_index: Optional[int],
        rng: np.random.Generator,
    ) -> Tuple[Optional[str], Optional[int]]:
        pool_by_task_index = self._load_task_to_episode_pool()
        if pool_by_task_index is None:
            return None, None

        repo_id = self._get_repo_id()
        task_index = self._resolve_task_index(frame)
        if task_index is None:
            _warn_annotated_task(
                repo_id,
                "sample_annotated_missing_task_index",
                "Sampled annotated-task loading enabled for repo '%s', but task_index could not be resolved. "
                "Falling back to per-episode annotated tasks.",
                repo_id,
            )
            return None, None

        episode_pool = pool_by_task_index.get(task_index)
        if episode_pool is None or len(episode_pool) == 0:
            _warn_annotated_task(
                repo_id,
                "sample_annotated_missing_task_pool",
                "Sampled annotated-task loading enabled for repo '%s', but task_index=%d is missing from %s. "
                "Falling back to per-episode annotated tasks.",
                repo_id,
                task_index,
                _TASK_TO_EPISODE_FILENAME,
            )
            return None, None

        sampled_episode_index = int(episode_pool[_sample_rng_index(rng, len(episode_pool))])
        sampled_task, is_frozen = self._resolve_annotated_task_for_episode_index(
            sampled_episode_index,
            missing_row_warning_kind="sample_annotated_missing_episode_row",
            missing_row_message=(
                "Sampled annotated-task loading enabled for repo '%s', but sampled episode_index=%d is missing from %s. "
                "Falling back to per-episode annotated tasks."
            ),
            invalid_type_warning_kind="sample_annotated_non_string_task",
            invalid_type_message=(
                "Sampled annotated-task loading enabled for repo '%s', but sampled episode_index=%d has a non-string task %r. "
                "Falling back to per-episode annotated tasks."
            ),
            blank_warning_kind="sample_annotated_blank_task",
            blank_message=(
                "Sampled annotated-task loading enabled for repo '%s', but sampled episode_index=%d has an empty task. "
                "Falling back to per-episode annotated tasks."
            ),
        )
        if is_frozen:
            _warn_annotated_task(
                repo_id,
                "sample_annotated_frozen_episode",
                "Sampled annotated-task loading enabled for repo '%s', but sampled episode_index=%d uses task=%r. "
                "Falling back to per-episode annotated tasks.",
                repo_id,
                sampled_episode_index,
                _FROZEN_TASK_SENTINEL,
            )
            return None, None

        current_episode_index = self._resolve_episode_index(frame, mapped_index)
        if sampled_task is None:
            return None, None
        if current_episode_index is not None and sampled_episode_index == current_episode_index:
            return sampled_task, None
        return sampled_task, sampled_episode_index

    def _resolve_annotated_task_for_frame(
        self,
        frame: Dict[str, Any],
        mapped_index: Optional[int],
        rng: np.random.Generator,
    ) -> Tuple[Optional[str], bool, Optional[int]]:
        if self.sample_annotated_task:
            sampled_task, sampled_episode_index = self._sample_annotated_task(frame, mapped_index, rng)
            if sampled_task is not None:
                return sampled_task, False, sampled_episode_index
        annotated_task, is_frozen = self._resolve_annotated_task(frame, mapped_index)
        return annotated_task, is_frozen, None

    def _resolve_base_camera_keys(
        self,
        example: Dict[str, Any],
        rng: np.random.Generator,
    ) -> List[str]:
        if self.camera_keys:
            if self.camera_keys_alternative and str(self.split).strip().lower() == "train":
                camera_key_configs = [self.camera_keys, self.camera_keys_alternative]
                selected_config_idx = _sample_rng_index(rng, len(camera_key_configs))
                return list(camera_key_configs[selected_config_idx])
            return list(self.camera_keys)
        if self._meta_camera_keys:
            return list(self._meta_camera_keys)
        return sorted(key for key in example if key.startswith("observation.images"))

    def _episode_camera_permutation(
        self,
        base_camera_keys: Sequence[str],
        episode_index: int,
    ) -> List[int]:
        if len(base_camera_keys) <= 1:
            return list(range(len(base_camera_keys)))
        seed_material = "\n".join(
            [
                self._get_repo_id(),
                str(self.random_camera_order_seed),
                str(int(episode_index)),
                *[str(key) for key in base_camera_keys],
            ]
        )
        digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
        permutation_seed = int.from_bytes(digest[:8], "little", signed=False)
        episode_rng = np.random.default_rng(permutation_seed)
        return [int(idx) for idx in episode_rng.permutation(len(base_camera_keys)).tolist()]

    def _resolve_camera_order(
        self,
        example: Dict[str, Any],
        rng: np.random.Generator,
        mapped_index: Optional[int],
    ) -> Tuple[List[str], List[int]]:
        base_camera_keys = self._resolve_base_camera_keys(example, rng)
        if not base_camera_keys:
            raise ValueError(
                f"No camera keys available for repo '{self._get_repo_id()}'. "
                "Provide camera_keys in tag metadata or ensure the dataset exposes observation.images* keys."
            )
        permutation = list(range(len(base_camera_keys)))
        if len(base_camera_keys) <= 1 or self.random_camera_order == "none":
            return base_camera_keys, permutation
        if self.random_camera_order == "all":
            permutation = [int(idx) for idx in rng.permutation(len(base_camera_keys)).tolist()]
            return [base_camera_keys[idx] for idx in permutation], permutation
        episode_index = self._resolve_episode_index(example, mapped_index)
        if episode_index is None:
            if not self._warned_camera_order_episode_fallback:
                log.warning(
                    "Unable to resolve episode_index for repo '%s'; falling back to fixed camera order.",
                    self._get_repo_id(),
                )
                self._warned_camera_order_episode_fallback = True
            return base_camera_keys, permutation
        permutation = self._episode_camera_permutation(base_camera_keys, episode_index)
        return [base_camera_keys[idx] for idx in permutation], permutation

    def _extract_question(self, example: Dict[str, Any]) -> str:
        keys: List[Optional[str]] = []
        if self.use_annotated_task and "task" in example:
            keys.append("task")
        keys.extend(
            [
                self.question_key,
                "language_instruction",
                "task",
            ]
        )
        seen_keys = set()
        for key in keys:
            if not key:
                continue
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if key not in example:
                continue
            value = example[key]
            if torch.is_tensor(value):
                value = value.item() if value.ndim == 0 else value.tolist()
            if isinstance(value, bytes):
                try:
                    text = value.decode("utf-8")
                except Exception:
                    text = str(value)
                normalized = _normalize_question_text(text)
                if normalized:
                    return normalized
                continue
            if isinstance(value, (list, tuple)):
                joined = " ".join(str(x) for x in value if x is not None)
                if joined:
                    normalized = _normalize_question_text(joined)
                    if normalized:
                        return normalized
            elif value is not None:
                normalized = _normalize_question_text(str(value))
                if normalized:
                    return normalized
        return ""

    def _extract_image(
        self,
        example: Dict[str, Any],
        keys_to_use: Sequence[str],
    ) -> Union[np.ndarray, List[np.ndarray]]:
        all_images: List[np.ndarray] = []
        available_keys = [key for key in example if key.startswith("observation.images")]
        if not keys_to_use:
            raise ValueError("No camera keys available in the LeRobot dataset")
        missing_keys = [key for key in keys_to_use if key not in example]
        if missing_keys:
            raise ValueError(
                f"Missing required camera keys {missing_keys} for repo '{self._get_repo_id()}'. "
                f"Available keys: {available_keys}"
            )
        for key in keys_to_use:
            frames = _collect_image_frames(example[key])
            all_images.extend(frames)
        if not all_images:
            raise ValueError("Failed to extract any camera images from LeRobot example")
        return all_images[0] if len(all_images) == 1 else all_images

    def _extract_state_action(
        self,
        example: Dict[str, Any],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Sequence[str]]]:
        state, state_keys_used = _extract_vector(example, self.state_keys, "observation.state.")
        action, action_keys_used = _extract_vector(example, self.action_keys, "action.")
        metadata: Dict[str, Sequence[str]] = {}
        if state_keys_used:
            metadata["state_keys"] = state_keys_used
        if action_keys_used:
            metadata["action_keys"] = action_keys_used
        return state, action, metadata

    def _build_episode_index(self) -> None:
        if self.drop_n_last_frames <= 0:
            return
        meta = getattr(self.dataset, "meta", None)
        if meta is None or not hasattr(meta, "episodes"):
            log.warning("LeRobot dataset metadata missing episodes; drop_n_last_frames ignored.")
            return
        episode_ids = (
            list(self.dataset.episodes)
            if getattr(self.dataset, "episodes", None) is not None
            else list(range(getattr(meta, "total_episodes", len(meta.episodes))))
        )
        absolute_to_relative = getattr(self.dataset, "_absolute_to_relative_idx", None)
        ranges: List[Tuple[int, int]] = []
        cumulative: List[int] = []
        total = 0
        for ep_idx in episode_ids:
            try:
                ep = meta.episodes[ep_idx]
            except Exception:
                continue
            start_abs = int(ep["dataset_from_index"])
            end_abs = int(ep["dataset_to_index"])
            trimmed_end_abs = max(start_abs, end_abs - self.drop_n_last_frames)
            start = start_abs
            trimmed_end = trimmed_end_abs
            if absolute_to_relative is not None and trimmed_end_abs > start_abs:
                rel_start = absolute_to_relative.get(start_abs)
                rel_last = absolute_to_relative.get(trimmed_end_abs - 1)
                if rel_start is None or rel_last is None:
                    log.warning(
                        "Skipping episode %s for drop_n_last_frames because absolute frame range [%s, %s) "
                        "is not present in the filtered dataset for repo '%s'.",
                        ep_idx,
                        start_abs,
                        trimmed_end_abs,
                        self._get_repo_id(),
                    )
                    continue
                start = int(rel_start)
                trimmed_end = int(rel_last) + 1
            if trimmed_end <= start:
                continue
            total += trimmed_end - start
            ranges.append((start, trimmed_end))
            cumulative.append(total)
        if ranges:
            self._episode_ranges = ranges
            self._episode_cumulative = cumulative
            self._effective_len = total
        else:
            log.warning("drop_n_last_frames removed all samples; falling back to full dataset.")

    def _map_index(self, item: int) -> int:
        if self._episode_ranges is None or self._episode_cumulative is None:
            return item
        if item < 0 or item >= self._episode_cumulative[-1]:
            raise IndexError(f"Index {item} out of bounds for filtered dataset.")
        pos = bisect_right(self._episode_cumulative, item)
        prev = self._episode_cumulative[pos - 1] if pos > 0 else 0
        start, _ = self._episode_ranges[pos]
        return start + (item - prev)

    def _get_frame_with_retry(
        self,
        item: int,
        rng: np.random.Generator,
        *,
        allow_random_retry: bool = True,
    ) -> Tuple[Dict[str, Any], int, int, Optional[int]]:
        self._ensure_dataset_loaded()
        max_retries = _get_env_int("LEROBOT_MAX_SKIP_RETRIES", 5)
        last_error: Optional[Exception] = None
        num_attempts = max_retries + 1 if allow_random_retry else 1
        for attempt in range(num_attempts):
            idx = item if attempt == 0 else int(rng.randint(0, len(self)))
            mapped_idx = self._map_index(idx)
            try:
                frame = self.dataset[mapped_idx]
            except Exception as exc:
                last_error = exc
                if not _should_skip_decode_error(exc):
                    raise
                log.warning(
                    "Skipping LeRobot example %s (mapped=%s, repo=%s) due to decode error (%s).",
                    idx,
                    mapped_idx,
                    self._get_repo_id(),
                    exc,
                )
                continue

            resolved_mapped_idx = frame.get("index")
            if torch.is_tensor(resolved_mapped_idx):
                resolved_mapped_idx = int(resolved_mapped_idx.item())
            elif resolved_mapped_idx is not None:
                resolved_mapped_idx = int(resolved_mapped_idx)
            else:
                resolved_mapped_idx = mapped_idx

            annotated_task, is_frozen, annotated_task_episode_index = self._resolve_annotated_task_for_frame(
                frame,
                resolved_mapped_idx,
                rng,
            )
            if is_frozen:
                if attempt >= num_attempts - 1:
                    repo_id = self._get_repo_id()
                    raise MalformedExampleError(
                        reason="annotated_all_frozen_frames",
                        details=(
                            "Failed to sample a non-frozen annotated LeRobot example for repo "
                            f"'{repo_id}' after {num_attempts} attempt(s): encountered task "
                            f"sentinel '{_FROZEN_TASK_SENTINEL}'."
                        ),
                        metadata={
                            "dataset_name": f"lerobot:{repo_id}",
                            "repo_id": repo_id,
                            "requested_item": int(item),
                            "resolved_mapped_index": int(resolved_mapped_idx),
                            "num_attempts": int(num_attempts),
                        },
                    )
                _warn_annotated_task(
                    self._get_repo_id(),
                    "annotated_frozen_resample",
                    "Skipping frozen annotated LeRobot samples for repo '%s' when task=%r.",
                    self._get_repo_id(),
                    _FROZEN_TASK_SENTINEL,
                )
                continue

            if annotated_task is not None:
                frame = dict(frame)
                frame["task"] = annotated_task
            return frame, mapped_idx, resolved_mapped_idx, annotated_task_episode_index
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to load any LeRobot example.")

    def _resolve_real_action_dim(
        self,
        action: np.ndarray,
        repo_id: str,
    ) -> int:
        observed_dim = int(action.shape[-1])
        stats_dim = self.robot_processor.get_action_dim(repo_id) if self.robot_processor is not None else None
        if stats_dim is not None and int(stats_dim) != observed_dim:
            raise ValueError(
                "LeRobot action dimension does not match robot stats: "
                f"repo_id={repo_id!r}, observed_action_dim={observed_dim}, stats_action_dim={int(stats_dim)}."
            )
        real_dim = observed_dim if stats_dim is None else int(stats_dim)
        if self.max_action_dim is not None and real_dim > self.max_action_dim:
            raise ValueError(
                f"Resolved action_dim {real_dim} exceeds configured max_action_dim {self.max_action_dim} "
                f"for repo_id={repo_id!r}."
            )
        return real_dim

    def _pad_action_to_max_dim(
        self,
        action: Optional[np.ndarray],
        repo_id: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if action is None:
            return None, None
        if action.ndim == 0:
            raise ValueError(f"Expected action tensor with at least 1 dimension, got shape {action.shape}.")
        real_dim = self._resolve_real_action_dim(action, repo_id)
        target_dim = real_dim if self.max_action_dim is None else self.max_action_dim
        action_dim_is_pad = np.ones((target_dim,), dtype=np.bool_)
        action_dim_is_pad[:real_dim] = False
        if target_dim == int(action.shape[-1]):
            return action, action_dim_is_pad
        padded_shape = list(action.shape)
        padded_shape[-1] = target_dim
        padded = np.zeros(padded_shape, dtype=action.dtype)
        padded[..., :real_dim] = action
        return padded, action_dim_is_pad

    def get(
        self,
        item: int,
        rng: np.random.Generator,
        *,
        normalize_state_action: bool = True,
        allow_random_retry: bool = True,
    ) -> Dict[str, Any]:
        frame, dataset_row_index, resolved_mapped_index, annotated_task_episode_index = self._get_frame_with_retry(
            item,
            rng,
            allow_random_retry=allow_random_retry,
        )
        mapped_index = frame.get("index")
        if torch.is_tensor(mapped_index):
            mapped_index = int(mapped_index.item())
        elif mapped_index is not None:
            mapped_index = int(mapped_index)
        else:
            mapped_index = resolved_mapped_index
        camera_keys_used, camera_permutation = self._resolve_camera_order(frame, rng, mapped_index)
        image = self._extract_image(frame, camera_keys_used)
        resize = _parse_image_resize(os.environ.get("LEROBOT_IMAGE_RESIZE"))
        if resize is not None:
            image = _resize_images(image, resize)
        question = self._extract_question(frame)
        if not question and not self._warned_missing_question:
            log.warning("LeRobot example is missing a language instruction; using an empty question string")
            self._warned_missing_question = True
        state, action, field_metadata = self._extract_state_action(frame)
        repo_id = self._get_repo_id()
        if normalize_state_action and self.robot_processor is not None:
            state = self.robot_processor.normalize_state(state, repo_id)
            action = self.robot_processor.normalize_action(action, repo_id)
        discrete_action_source = action
        action_time_is_pad = None
        action_dim_is_pad = None
        if self.action_format in {"continuous", "both"}:
            action, action_time_is_pad = _pad_action_to_max_horizon(
                action,
                tag_action_horizon=self.tag_action_horizon,
                max_action_horizon=self.max_action_horizon,
            )
            action, action_dim_is_pad = self._pad_action_to_max_dim(action, repo_id)
        if state is None and not self._warned_missing_state:
            log.warning("LeRobot example does not contain any state vectors")
            self._warned_missing_state = True
        if action is None and not self._warned_missing_action:
            log.warning("LeRobot example does not contain any action vectors")
            self._warned_missing_action = True
        discrete_state_string = ""
        if self.state_format in {"discrete", "both"}:
            discrete_state_string = build_discrete_state_string(
                state,
                num_state_tokens=self.num_discrete_state_tokens,
            )
        sampled_style = self._sample_style(rng)
        emit_depth = style_uses_depth_output(sampled_style)
        emit_action = style_uses_action_output(sampled_style)
        if emit_depth and not self.enable_depth_reasoning:
            raise ValueError(
                f"Style '{sampled_style}' requires enable_depth_reasoning=True."
            )

        discrete_action_string = ""
        if self.action_format in {"discrete", "both"}:
            discrete_action_string = build_discrete_action_string_from_action(
                discrete_action_source,
                self._get_action_discrete_processor(),
                max_token_id=self.max_discrete_action_token_id,
                on_out_of_range=self._warn_discrete_action_oob,
            )
        if not emit_action:
            discrete_action_string = ""
        task_progress = self._compute_task_progress(frame, mapped_index)
        depth_buffer_codes = None
        depth_updated_mask = None
        depth_camera_key_used = None
        if emit_depth:
            depth_camera_key_used, selected_depth_dataset = self._select_depth_dataset_for_cameras(camera_keys_used)
            if selected_depth_dataset is not None:
                try:
                    # Depth companion datasets are indexed by loaded row position, which can differ
                    # from frame["index"] when parquet rows are missing relative to metadata.
                    depth_frame = selected_depth_dataset[dataset_row_index]
                    depth_buffer_codes = depth_frame.get("depth_buffer_codes")
                    if depth_buffer_codes is None:
                        depth_buffer_codes = depth_frame.get("buffer_codes")
                    depth_updated_mask = depth_frame.get("depth_updated_mask")
                except (IndexError, KeyError):
                    depth_buffer_codes = None
                    depth_updated_mask = None
            if depth_buffer_codes is None:
                depth_buffer_codes = frame.get("depth_buffer_codes")
                if depth_buffer_codes is None:
                    depth_buffer_codes = frame.get("buffer_codes")
                depth_updated_mask = frame.get("depth_updated_mask")
        discrete_depth_string = ""
        if emit_depth:
            if depth_buffer_codes is None:
                raise ValueError(
                    "Depth output style requires frame['depth_buffer_codes'] (or legacy frame['buffer_codes'])."
                )
            depth_buffer_codes = _validate_depth_side_channel_length(
                depth_buffer_codes,
                expected=self.num_depth_tokens_per_image,
                label="depth_buffer_codes",
            ).astype(np.int64, copy=False)
            if depth_updated_mask is not None:
                depth_updated_mask = _validate_depth_side_channel_length(
                    depth_updated_mask,
                    expected=self.num_depth_tokens_per_image,
                    label="depth_updated_mask",
                ).astype(np.bool_, copy=False)
            discrete_depth_string = build_discrete_depth_string(
                depth_buffer_codes,
                num_depth_tokens=self.num_depth_tokens,
            )
        discrete_answer_string = f"{discrete_depth_string}{discrete_action_string}"

        metadata: Dict[str, Any] = dict(field_metadata)
        metadata["repo_id"] = repo_id
        metadata["split"] = self.split
        metadata["camera_keys_used"] = list(camera_keys_used)
        metadata["camera_permutation"] = list(camera_permutation)
        metadata["depth_camera_key_used"] = depth_camera_key_used
        metadata["observation_delta_indices"] = list(self.observation_delta_indices)
        metadata["action_delta_indices"] = list(self.action_delta_indices)
        metadata["action_horizon"] = int(self.tag_action_horizon)
        metadata["n_action_steps"] = int(self.tag_n_action_steps)
        if "timestamp" in frame:
            timestamp = frame["timestamp"]
            if torch.is_tensor(timestamp):
                timestamp = timestamp.item()
            metadata["timestamp"] = float(timestamp)
        if "episode_index" in frame:
            episode_index = frame["episode_index"]
            if torch.is_tensor(episode_index):
                episode_index = episode_index.item()
            metadata["episode_index"] = int(episode_index)
        if "frame_index" in frame:
            frame_index = frame["frame_index"]
            if torch.is_tensor(frame_index):
                frame_index = frame_index.item()
            metadata["frame_index"] = int(frame_index)
        if "task" in frame:
            metadata["task"] = frame["task"] if isinstance(frame["task"], str) else str(frame["task"])
        if annotated_task_episode_index is not None:
            metadata["annotated_task_episode_index"] = int(annotated_task_episode_index)
        metadata["task_progress"] = float(task_progress)

        if sampled_style in ROBOT_OUTPUT_STYLES:
            tag_metadata = (
                self.robot_processor_config.tag_metadata.get(repo_id)
                if self.robot_processor_config is not None and getattr(self.robot_processor_config, "tag_metadata", None)
                else {}
            )
            if not tag_metadata and self.robot_processor_config is not None:
                resolved_tag = self.robot_processor_config.repo_to_tag.get(repo_id, repo_id)
                tag_metadata = self.robot_processor_config.tag_metadata.get(resolved_tag, {})
            setup_type = wrap_setup_text(
                str(tag_metadata.get("setup_type", "")),
                add_setup_tokens=self.data_formatter_add_setup_tokens,
            )
            control_mode = wrap_control_text(
                str(tag_metadata.get("control_mode", "")),
                add_control_tokens=self.data_formatter_add_control_tokens,
            )
            prompt_fields = build_robot_prompt_fields(
                question,
                style=sampled_style,
                discrete_state_string=discrete_state_string,
                setup_type=setup_type,
                control_mode=control_mode,
            )
        else:
            prompt_text = question
            if discrete_state_string:
                prompt_text = append_discrete_state_to_prompt(prompt_text, discrete_state_string)
            prompt_fields = {"question": prompt_text, "style": sampled_style}

        if self.action_format == "continuous":
            # Prompt-only text conditioning for continuous-flow training.
            text_fields = {"messages": {**prompt_fields, "prompt_only": True}}
        else:
            # Discrete-token supervision via answer string.
            text_fields = {
                **prompt_fields,
                "answers": discrete_answer_string,
            }

        example_out: Dict[str, Any] = {
            "image": image,
            **text_fields,
            "metadata": metadata,
        }
        if self.state_format in {"continuous", "both"}:
            example_out["state"] = state
        if self.action_format in {"continuous", "both"} and emit_action:
            example_out["action"] = action
        if self.action_format in {"continuous", "both"} and emit_action and action_time_is_pad is not None:
            example_out["action_horizon_is_pad"] = action_time_is_pad
        if self.action_format in {"continuous", "both"} and emit_action and action_dim_is_pad is not None:
            example_out["action_dim_is_pad"] = action_dim_is_pad
        if emit_depth and depth_updated_mask is not None:
            example_out["depth_updated_mask"] = np.asarray(depth_updated_mask, dtype=np.bool_)
        if emit_depth and depth_buffer_codes is not None:
            example_out["depth_buffer_codes"] = np.asarray(depth_buffer_codes, dtype=np.int64)
        return example_out


def build_lerobot_dataset(
    dataset_name: str,
    split: str,
    *,
    n_obs_steps: Optional[int] = None,
    action_horizon: Optional[int] = None,
    max_action_horizon: Optional[int] = None,
    max_action_dim: Optional[int] = None,
    drop_n_last_frames: Optional[int] = None,
    random_camera_order: Optional[str] = None,
    random_camera_order_seed: Optional[int] = None,
    use_annotated_task: Optional[bool] = None,
    sample_annotated_task: Optional[bool] = None,
) -> LeRobotDatasetWrapper:
    prefix = "lerobot:"
    if not dataset_name.startswith(prefix):
        raise ValueError(f"Invalid LeRobot dataset name '{dataset_name}'")
    repo_spec = dataset_name[len(prefix):]
    if not repo_spec:
        raise ValueError("Dataset name must include a repo id")

    parsed = _parse_repo_spec(repo_spec)
    metadata_repo_id = _canonicalize_depth_repo_id(parsed.repo_id)
    root_base = os.environ.get("LEROBOT_DATA_ROOT")
    download_videos = True
    question_key = "task"
    style = "demo"
    style_sampling_rates = _normalize_style_sampling_rates(_get_env_json("LEROBOT_STYLE_SAMPLING_RATES"))

    n_obs_steps = _resolve_required_positive_int(
        n_obs_steps,
        env_name="LEROBOT_N_OBS_STEPS",
        label="n_obs_steps",
    )
    if max_action_horizon is not None and action_horizon is not None:
        raise ValueError("Specify only one of action_horizon or max_action_horizon.")
    resolved_max_action_horizon = max_action_horizon if max_action_horizon is not None else action_horizon
    raw_env_action_horizon = os.environ.get("LEROBOT_MAX_ACTION_HORIZON")
    if (raw_env_action_horizon is None or not raw_env_action_horizon.strip()) and resolved_max_action_horizon is None:
        raw_env_action_horizon = os.environ.get("LEROBOT_ACTION_HORIZON")
    if resolved_max_action_horizon is None:
        if raw_env_action_horizon is None or not raw_env_action_horizon.strip():
            raise ValueError(
                "max_action_horizon must be provided or LEROBOT_MAX_ACTION_HORIZON must be set."
            )
        resolved_max_action_horizon = int(raw_env_action_horizon)
    max_action_horizon = _require_positive_int(
        int(resolved_max_action_horizon),
        label="max_action_horizon",
    )
    if max_action_dim is None:
        raw_max_action_dim = os.environ.get("LEROBOT_MAX_ACTION_DIM")
        if raw_max_action_dim is not None and raw_max_action_dim.strip():
            max_action_dim = _require_positive_int(int(raw_max_action_dim), label="max_action_dim")
    else:
        max_action_dim = _require_positive_int(max_action_dim, label="max_action_dim")
    if drop_n_last_frames is None:
        drop_n_last_frames = 0
    else:
        drop_n_last_frames = _require_non_negative_int(drop_n_last_frames, label="drop_n_last_frames")
    video_backend = _get_env_str("LEROBOT_VIDEO_BACKEND", "pyav")
    tolerance_s = _get_env_float("LEROBOT_TOLERANCE_S", 1e-3)
    action_format = _get_required_env_choice(
        "LEROBOT_ACTION_FORMAT",
        choices=sorted(SUPPORTED_ACTION_FORMATS),
    )
    state_format = _get_required_env_choice(
        "LEROBOT_STATE_FORMAT",
        choices=sorted(SUPPORTED_STATE_FORMATS),
    )
    num_discrete_state_tokens = _get_env_int(
        "LEROBOT_NUM_STATE_TOKENS",
        DEFAULT_NUM_STATE_TOKENS,
    )
    enable_depth_reasoning = _get_env_bool("LEROBOT_ENABLE_DEPTH_REASONING", False)
    add_depth_tokens = _get_env_bool("LEROBOT_ADD_DEPTH_TOKENS", False)
    num_depth_tokens = _get_env_int(
        "LEROBOT_NUM_DEPTH_TOKENS",
        DEFAULT_NUM_DEPTH_TOKENS,
    )
    num_depth_tokens_per_image = _get_env_int(
        "LEROBOT_NUM_DEPTH_TOKENS_PER_IMAGE",
        100,
    )
    if num_depth_tokens_per_image <= 0:
        raise ValueError(
            f"LEROBOT_NUM_DEPTH_TOKENS_PER_IMAGE must be > 0, got {num_depth_tokens_per_image}."
        )
    discrete_action_tokenizer = _get_env_str("LEROBOT_DISCRETE_ACTION_TOKENIZER", "").strip()
    if action_format in {"discrete", "both"} and not discrete_action_tokenizer:
        raise ValueError(
            "Discrete LeRobot action formatting requires LEROBOT_DISCRETE_ACTION_TOKENIZER to be set."
        )
    if not discrete_action_tokenizer:
        discrete_action_tokenizer = None
    raw_num_action_tokens = os.environ.get("LEROBOT_NUM_ACTION_TOKENS")
    max_discrete_action_token_id: Optional[int] = None
    if raw_num_action_tokens is not None and raw_num_action_tokens.strip():
        try:
            max_discrete_action_token_id = int(raw_num_action_tokens)
        except ValueError:
            log.warning(
                "Invalid LEROBOT_NUM_ACTION_TOKENS='%s', ignoring token-id range checks.",
                raw_num_action_tokens,
            )

    use_depth_dataset = any(
        rate > 0.0 and style_uses_depth_output(style_name)
        for style_name, rate in (style_sampling_rates or {}).items()
    )
    if use_depth_dataset and not enable_depth_reasoning:
        raise ValueError("Depth output styles require LEROBOT_ENABLE_DEPTH_REASONING=1.")
    dataset_repo_id = parsed.repo_id

    dataset_root = None
    if root_base:
        base_path = Path(root_base).expanduser()
        # Always scope to requested repo_id path. If missing locally,
        # LeRobotDataset will download into this location.
        dataset_root = base_path / dataset_repo_id

    depth_dataset: Optional[LeRobotDataset] = None
    depth_dataset_root: Optional[Path] = None
    depth_dataset_repo_id = dataset_repo_id
    depth_dataset_reopen_kwargs: Optional[Dict[str, Any]] = None
    if enable_depth_reasoning and use_depth_dataset:
        depth_root_base = os.environ.get("LEROBOT_DEPTH_DATA_ROOT")
        if depth_root_base:
            depth_dataset_root = _resolve_depth_dataset_root(depth_root_base, dataset_repo_id)
        else:
            if not dataset_repo_id.endswith(_DEPTH_DATASET_SUFFIX):
                depth_dataset_repo_id = f"{dataset_repo_id}{_DEPTH_DATASET_SUFFIX}"
            if root_base:
                depth_dataset_root = Path(root_base).expanduser() / depth_dataset_repo_id

    depth_dataset_reopen_kwargs_by_camera: Dict[str, Dict[str, Any]] = {}
    depth_root_by_camera_env = _get_env_json("LEROBOT_DEPTH_DATA_ROOT_BY_CAMERA")
    if enable_depth_reasoning and use_depth_dataset and depth_root_by_camera_env is not None:
        if not isinstance(depth_root_by_camera_env, dict):
            raise ValueError(
                "LEROBOT_DEPTH_DATA_ROOT_BY_CAMERA must be a JSON object mapping camera keys "
                "to depth dataset root directories."
            )
        for camera_key, depth_root_base_for_camera in depth_root_by_camera_env.items():
            camera_key = str(camera_key).strip()
            if not camera_key:
                raise ValueError("LEROBOT_DEPTH_DATA_ROOT_BY_CAMERA contains an empty camera key.")
            if not isinstance(depth_root_base_for_camera, str) or not depth_root_base_for_camera.strip():
                raise ValueError(
                    "LEROBOT_DEPTH_DATA_ROOT_BY_CAMERA values must be non-empty root directory strings."
                )
            depth_dataset_root_for_camera = _resolve_depth_dataset_root(
                depth_root_base_for_camera,
                dataset_repo_id,
            )
            depth_dataset_reopen_kwargs_by_camera[camera_key] = dict(
                repo_id=dataset_repo_id,
                root=str(depth_dataset_root_for_camera),
                episodes=parsed.episodes,
                download_videos=False,
                video_backend=video_backend,
                tolerance_s=tolerance_s,
                revision="main",
            )

    repo_to_tag_env = _get_env_json("LEROBOT_REPO_TO_TAG") or {}
    tag_metadata_env = _get_env_json("LEROBOT_TAG_METADATA")
    random_camera_order = _normalize_random_camera_order(
        random_camera_order
        if random_camera_order is not None
        else _get_env_str("LEROBOT_RANDOM_CAMERA_ORDER", "none"),
        label="random_camera_order",
    )
    random_camera_order_seed = _require_non_negative_int(
        random_camera_order_seed
        if random_camera_order_seed is not None
        else _get_env_non_negative_int("LEROBOT_RANDOM_CAMERA_ORDER_SEED", 0),
        label="random_camera_order_seed",
    )
    tag_metadata = _require_tag_metadata_entry(
        tag_metadata_env,
        repo_to_tag_env,
        metadata_repo_id,
        random_camera_order=random_camera_order,
    )
    tag = str(repo_to_tag_env[metadata_repo_id])
    tag_action_horizon = _require_positive_int(
        int(tag_metadata["action_horizon"]),
        label="tag_action_horizon",
    )
    tag_n_action_steps = _require_positive_int(
        int(tag_metadata["n_action_steps"]),
        label="tag_n_action_steps",
    )
    if tag_n_action_steps > tag_action_horizon:
        raise ValueError(
            f"Tag metadata for repo '{metadata_repo_id}' has n_action_steps={tag_n_action_steps}, "
            f"which exceeds action_horizon={tag_action_horizon}."
        )
    if tag_action_horizon > max_action_horizon:
        raise ValueError(
            f"Tag metadata for repo '{metadata_repo_id}' has action_horizon={tag_action_horizon}, "
            f"which exceeds configured max_action_horizon={max_action_horizon}."
        )

    delta_timestamps, obs_indices, action_indices = _resolve_delta_settings(
        parsed.repo_id,
        dataset_root,
        n_obs_steps,
        tag_action_horizon,
    )

    def _build_dataset() -> LeRobotDataset:
        dataset_kwargs = dict(
            repo_id=dataset_repo_id,
            root=str(dataset_root) if dataset_root else None,
            episodes=parsed.episodes,
            download_videos=download_videos,
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
            tolerance_s=tolerance_s,
        )
        try:
            return _MolmoLeRobotDataset(**dataset_kwargs)
        except Exception as exc:
            if not _should_retry_with_main_revision(exc):
                raise
            log.warning(
                "LeRobot dataset init failed for %s (%s). Retrying with revision='main'.",
                dataset_repo_id,
                exc,
            )
            return _MolmoLeRobotDataset(**dataset_kwargs, revision="main")

    dataset = _build_dataset_with_lock(dataset_root, _build_dataset)

    if enable_depth_reasoning and use_depth_dataset and depth_dataset_root is not None:
        depth_build_kwargs = dict(
            repo_id=depth_dataset_repo_id,
            root=str(depth_dataset_root),
            episodes=parsed.episodes,
            download_videos=False,
            video_backend=video_backend,
            tolerance_s=tolerance_s,
            revision="main",
        )

        def _build_depth_dataset() -> LeRobotDataset:
            import huggingface_hub.constants as _hf_consts

            previous_hf_offline = _hf_consts.HF_HUB_OFFLINE
            _hf_consts.HF_HUB_OFFLINE = True
            try:
                try:
                    return _MolmoLeRobotDataset(**depth_build_kwargs)
                except Exception as exc:
                    if not _should_retry_with_main_revision(exc):
                        raise
                    log.warning(
                        "LeRobot depth dataset init failed for %s (%s). Retrying with revision='main'.",
                        depth_dataset_repo_id,
                        exc,
                    )
                    return _MolmoLeRobotDataset(**{**depth_build_kwargs, "revision": "main"})
            finally:
                _hf_consts.HF_HUB_OFFLINE = previous_hf_offline

        try:
            depth_dataset = _build_depth_dataset()
            depth_dataset_reopen_kwargs = dict(
                depth_build_kwargs,
                revision=getattr(depth_dataset, "revision", None),
            )
        except Exception as exc:
            log.warning(
                "Failed to build depth companion dataset for %s from %s (%s). "
                "Falling back to main dataset for depth fields.",
                depth_dataset_repo_id,
                depth_dataset_root,
                exc,
            )
            depth_dataset = None
            depth_dataset_reopen_kwargs = None

    stats_by_tag_env = _get_env_json("LEROBOT_STATS_BY_TAG")
    norm_mode = _get_env_str("LEROBOT_NORM_MODE", "min_max")
    norm_clip = _get_env_bool("LEROBOT_NORM_CLIP", True)
    add_setup_tokens = _get_env_bool("LEROBOT_ADD_SETUP_TOKENS", False)
    add_control_tokens = _get_env_bool("LEROBOT_ADD_CONTROL_TOKENS", False)
    if use_annotated_task is None:
        use_annotated_task = _get_env_bool("LEROBOT_USE_ANNOTATED_TASK", False)
    if sample_annotated_task is None:
        sample_annotated_task = _get_env_bool("LEROBOT_SAMPLE_ANNOTATED_TASK", False)
    raw_camera_keys = tag_metadata.get("camera_keys")
    camera_keys = _normalize_optional_string_sequence(
        None if raw_camera_keys is None else [str(v) for v in raw_camera_keys],
        label="camera_keys",
    )
    raw_camera_keys_alternative = tag_metadata.get("camera_keys_alternative")
    camera_keys_alternative = _normalize_optional_string_sequence(
        None if raw_camera_keys_alternative is None else [str(v) for v in raw_camera_keys_alternative],
        label="camera_keys_alternative",
    )
    if camera_keys_alternative and not camera_keys:
        raise ValueError(
            f"Tag metadata for repo '{parsed.repo_id}' defines camera_keys_alternative without "
            "a primary camera_keys list."
        )
    state_keys = _require_non_empty_string_sequence(
        [str(v) for v in tag_metadata["state_keys"]],
        label="state_keys",
    )
    action_keys = _require_non_empty_string_sequence(
        [str(tag_metadata["action_key"])],
        label="action_keys",
    )
    available_cameras = list(getattr(dataset.meta, "camera_keys", []) or [])
    camera_keys = _resolve_repo_camera_keys(
        tag,
        available_cameras=available_cameras,
        camera_keys=camera_keys,
    )
    camera_keys_alternative = _resolve_repo_camera_keys(
        tag,
        available_cameras=available_cameras,
        camera_keys=camera_keys_alternative,
    )
    _validate_available_camera_keys(
        available_cameras,
        camera_keys,
        repo_id=parsed.repo_id,
        label="tag metadata camera_keys",
    )
    _validate_available_camera_keys(
        available_cameras,
        camera_keys_alternative,
        repo_id=parsed.repo_id,
        label="tag metadata camera_keys_alternative",
    )

    action_proc_cfg = None
    if isinstance(stats_by_tag_env, dict) and stats_by_tag_env:
        action_proc_cfg = RobotProcessorConfig.from_stats(
            stats_by_tag=stats_by_tag_env,
            tag_metadata=tag_metadata_env if isinstance(tag_metadata_env, dict) else {},
            repo_to_tag=repo_to_tag_env if isinstance(repo_to_tag_env, dict) else {},
            norm_mode=norm_mode,
            norm_clip=norm_clip,
            data_formatter_add_setup_tokens=add_setup_tokens,
            data_formatter_add_control_tokens=add_control_tokens,
        )
    else:
        raise ValueError(
            "LeRobot tagged stats are required. Set LEROBOT_STATS_BY_TAG, LEROBOT_REPO_TO_TAG, "
            "and LEROBOT_TAG_METADATA before building LeRobot datasets."
        )
    robot_processor = action_proc_cfg.build_processor() if action_proc_cfg else None

    wrapper = LeRobotDatasetWrapper(
        dataset=dataset,
        split=split,
        dataset_reopen_kwargs={
            "repo_id": dataset_repo_id,
            "root": str(dataset_root) if dataset_root else None,
            "episodes": parsed.episodes,
            "download_videos": download_videos,
            "delta_timestamps": delta_timestamps,
            "video_backend": video_backend,
            "tolerance_s": tolerance_s,
            "revision": getattr(dataset, "revision", None),
        },
        camera_keys=camera_keys,
        camera_keys_alternative=camera_keys_alternative,
        question_key=question_key,
        state_keys=state_keys,
        action_keys=action_keys,
        observation_indices=obs_indices,
        action_indices=action_indices,
        drop_n_last_frames=drop_n_last_frames,
        style=style,
        style_sampling_rates=style_sampling_rates,
        action_format=action_format,
        state_format=state_format,
        num_discrete_state_tokens=num_discrete_state_tokens,
        enable_depth_reasoning=enable_depth_reasoning,
        add_depth_tokens=add_depth_tokens,
        num_depth_tokens=num_depth_tokens,
        num_depth_tokens_per_image=num_depth_tokens_per_image,
        discrete_action_tokenizer=discrete_action_tokenizer,
        max_discrete_action_token_id=max_discrete_action_token_id,
        data_formatter_add_setup_tokens=add_setup_tokens,
        data_formatter_add_control_tokens=add_control_tokens,
        random_camera_order=random_camera_order,
        random_camera_order_seed=random_camera_order_seed,
        use_annotated_task=use_annotated_task,
        sample_annotated_task=sample_annotated_task,
        tag_action_horizon=tag_action_horizon,
        tag_n_action_steps=tag_n_action_steps,
        max_action_horizon=max_action_horizon,
        max_action_dim=max_action_dim,
        robot_processor=robot_processor,
        robot_processor_config=action_proc_cfg,
        metadata_repo_id=metadata_repo_id,
        depth_dataset=depth_dataset,
        depth_dataset_reopen_kwargs=depth_dataset_reopen_kwargs,
        depth_dataset_reopen_kwargs_by_camera=depth_dataset_reopen_kwargs_by_camera,
    )
    if use_annotated_task:
        wrapper.preload_annotated_tasks()
    return wrapper


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Quick test loader for LeRobot datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="lerobot:HuggingFaceVLA/libero",
        help="Dataset name following the lerobot:<user>/<repo> format.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to load (e.g., train/validation/test).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=1,
        help="Example index to preview.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed forwarded to the dataset RNG.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset

    print(f"Loading {dataset_name} ({args.split} split)")
    ds = build_lerobot_dataset(dataset_name, args.split)
    print(f"Loaded {len(ds)} examples from {ds.dataset.repo_id}")

    rng = np.random.default_rng(args.seed)
    example = ds.get(args.index % len(ds), rng)
    question = example.get("question")
    if question is None and "messages" in example:
        messages = example["messages"]
        if isinstance(messages, dict):
            msg_list = messages.get("messages", [])
            question = msg_list[0] if msg_list else ""
        elif isinstance(messages, list):
            question = messages[0] if messages else ""
    image_value = example["image"]
    if isinstance(image_value, list):
        image_num = len(image_value)
        image_shape = image_value[0].shape if image_value else None
    else:
        image_num = 1
        image_shape = image_value.shape
    action_value = example.get("action")
    print(
        f'Example {args.index}: question={question},',
        f'image_num={image_num},',
        f'image_shape={image_shape},',
        f'state_shape={None if example["state"] is None else tuple(example["state"].shape)},',
        f'action_shape={None if action_value is None else tuple(action_value.shape)}'
    )
    print(json.dumps(example["metadata"], indent=2))
    print(example.keys())


if __name__ == "__main__":
    _main()
