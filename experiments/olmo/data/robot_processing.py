import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, List

import numpy as np
import torch

from olmo.config import BaseConfig, D


def _to_array(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _to_mask(x, *, fallback_like: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if x is None:
        if fallback_like is None:
            return None
        return np.ones_like(fallback_like, dtype=np.bool_)
    mask = np.asarray(x, dtype=np.bool_)
    return mask


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _normalize_state_keys(
    value: Optional[Iterable[Any]],
    *,
    default: Sequence[str] = ("observation.state",),
) -> List[str]:
    if isinstance(value, str):
        raw_values = [value]
    else:
        raw_values = list(value) if value is not None else list(default)
    normalized = [str(key) for key in raw_values if str(key)]
    if not normalized:
        normalized = [str(key) for key in default if str(key)]
    if not normalized:
        raise ValueError("state_keys must contain at least one non-empty key.")
    return normalized


def _stats_reference_array(stats: Optional[Mapping[str, Any]]) -> Optional[np.ndarray]:
    if not isinstance(stats, Mapping):
        return None
    for key in ("mean", "std", "min", "max", "q01", "q99", "q10", "q90", "mask"):
        value = stats.get(key)
        if value is None:
            continue
        arr = _to_array(value)
        if arr is not None and getattr(arr, "shape", None):
            return arr
    return None


def _normalize_array(
    x: np.ndarray,
    *,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    min_val: Optional[np.ndarray] = None,
    max_val: Optional[np.ndarray] = None,
    q_low: Optional[np.ndarray] = None,
    q_high: Optional[np.ndarray] = None,
    mode: str = "mean_std",
) -> np.ndarray:
    eps = 1e-6
    if mode == "none":
        return x
    if mode == "mean_std":
        assert mean is not None and std is not None
        return (x - mean) / np.maximum(std, eps)
    if mode == "min_max":
        assert min_val is not None and max_val is not None
        denom = np.maximum(max_val - min_val, eps)
        return 2.0 * (x - min_val) / denom - 1.0
    if mode == "q01_q99":
        assert q_low is not None and q_high is not None
        denom = np.maximum(q_high - q_low, eps)
        return 2.0 * (x - q_low) / denom - 1.0
    if mode == "q10_q90":
        assert q_low is not None and q_high is not None
        denom = np.maximum(q_high - q_low, eps)
        return 2.0 * (x - q_low) / denom - 1.0
    return x


def _unnormalize_array(
    x: np.ndarray,
    *,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    min_val: Optional[np.ndarray] = None,
    max_val: Optional[np.ndarray] = None,
    q_low: Optional[np.ndarray] = None,
    q_high: Optional[np.ndarray] = None,
    mode: str = "mean_std",
) -> np.ndarray:
    if mode == "none":
        return x
    if mode == "mean_std":
        assert mean is not None and std is not None
        return x * std + mean
    if mode == "min_max":
        assert min_val is not None and max_val is not None
        return (x + 1.0) * (max_val - min_val) / 2.0 + min_val
    if mode in {"q01_q99", "q10_q90"}:
        assert q_low is not None and q_high is not None
        return (x + 1.0) * (q_high - q_low) / 2.0 + q_low
    return x


def _uses_bounded_normalized_range(mode: str) -> bool:
    return mode in {"min_max", "q01_q99", "q10_q90"}


def _feature_dim_from_normalizer(normalizer: Optional["_FeatureNormalizer"]) -> Optional[int]:
    if normalizer is None:
        return None
    for attr_name in ("mean", "std", "min_val", "max_val", "q_low", "q_high", "mask"):
        value = getattr(normalizer, attr_name, None)
        if value is None:
            continue
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) == 0:
            continue
        return int(shape[-1])
    return None


def _feature_dim_from_stats(stats: Optional[Mapping[str, Any]]) -> Optional[int]:
    if not isinstance(stats, Mapping):
        return None
    for key in ("mean", "std", "min", "max", "q01", "q99", "q10", "q90", "mask", "names"):
        value = stats.get(key)
        if value is None:
            continue
        arr = _to_array(value)
        if arr is not None:
            shape = getattr(arr, "shape", None)
            if shape is not None and len(shape) > 0:
                return int(shape[-1])
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return int(len(value))
    return None


def _raise_missing_stats(mode: str, stats: Mapping[str, Sequence[float]], required_keys: Sequence[str]) -> None:
    available = sorted(str(key) for key in stats.keys()) if isinstance(stats, Mapping) else []
    raise ValueError(
        f"norm_mode={mode!r} requires stats keys {list(required_keys)!r}, "
        f"but found only {available!r}."
    )


@dataclass
class _FeatureNormalizer:
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    min_val: Optional[np.ndarray] = None
    max_val: Optional[np.ndarray] = None
    q_low: Optional[np.ndarray] = None
    q_high: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    zero_mask: Optional[np.ndarray] = None
    mode: str = "min_max"
    #: Whether a bounded mode saturates the normalized range at [-1, 1]. On by
    #: default; off leaves values outside the fitted band in proportion, so the
    #: 1% tails of a q01_q99 fit train and serve as recorded instead of at the
    #: band's edge. Ignored by the unbounded modes, which never clipped.
    clip: bool = True

    @classmethod
    def from_stats(
        cls,
        stats: Mapping[str, Sequence[float]],
        mode: str,
        clip: bool = True,
    ) -> Optional["_FeatureNormalizer"]:
        normalizer = cls._from_stats(stats, mode)
        if normalizer is not None:
            normalizer.clip = bool(clip)
        return normalizer

    @classmethod
    def _from_stats(
        cls,
        stats: Mapping[str, Sequence[float]],
        mode: str,
    ) -> Optional["_FeatureNormalizer"]:
        if stats is None:
            return None
        raw_mask = stats.get("mask") if isinstance(stats, Mapping) else None
        if mode == "none":
            reference = _stats_reference_array(stats)
            mask = _to_mask(raw_mask, fallback_like=reference)
            return cls(mask=mask, mode=mode)
        if mode == "mean_std":
            mean = _to_array(stats.get("mean"))
            std = _to_array(stats.get("std"))
            if mean is None or std is None:
                _raise_missing_stats(mode, stats, ("mean", "std"))
            mask = _to_mask(raw_mask, fallback_like=mean)
            return cls(mean=mean, std=std, mask=mask, mode=mode)
        if mode == "min_max":
            min_val = _to_array(stats.get("min"))
            max_val = _to_array(stats.get("max"))
            if min_val is None or max_val is None:
                _raise_missing_stats(mode, stats, ("min", "max"))
            mask = _to_mask(raw_mask, fallback_like=min_val)
            zero_mask = (min_val == max_val)
            return cls(min_val=min_val, max_val=max_val, mask=mask, zero_mask=zero_mask, mode=mode)
        if mode == "q01_q99":
            q_low = _to_array(stats.get("q01"))
            q_high = _to_array(stats.get("q99"))
            if q_low is None or q_high is None:
                _raise_missing_stats(mode, stats, ("q01", "q99"))
            min_val = _to_array(stats.get("min"))
            max_val = _to_array(stats.get("max"))
            fallback = min_val if min_val is not None else q_low
            mask = _to_mask(raw_mask, fallback_like=fallback)
            zero_mask = None if min_val is None or max_val is None else (min_val == max_val)
            return cls(
                min_val=min_val,
                max_val=max_val,
                q_low=q_low,
                q_high=q_high,
                mask=mask,
                zero_mask=zero_mask,
                mode=mode,
            )
        if mode == "q10_q90":
            q_low = _to_array(stats.get("q10"))
            q_high = _to_array(stats.get("q90"))
            if q_low is None or q_high is None:
                _raise_missing_stats(mode, stats, ("q10", "q90"))
            min_val = _to_array(stats.get("min"))
            max_val = _to_array(stats.get("max"))
            fallback = min_val if min_val is not None else q_low
            mask = _to_mask(raw_mask, fallback_like=fallback)
            zero_mask = None if min_val is None or max_val is None else (min_val == max_val)
            return cls(
                min_val=min_val,
                max_val=max_val,
                q_low=q_low,
                q_high=q_high,
                mask=mask,
                zero_mask=zero_mask,
                mode=mode,
            )
        return None

    def normalize(self, x):
        arr = _to_array(x)
        if arr is None:
            return None
        normed = _normalize_array(
            arr,
            mean=self.mean,
            std=self.std,
            min_val=self.min_val,
            max_val=self.max_val,
            q_low=self.q_low,
            q_high=self.q_high,
            mode=self.mode,
        )
        if self.clip and _uses_bounded_normalized_range(self.mode):
            normed = np.clip(normed, -1.0, 1.0)
        if self.mask is not None:
            normed = np.where(self.mask, normed, arr)
        if self.zero_mask is not None:
            normed = np.where(self.zero_mask, 0.0, normed)
        if torch.is_tensor(x):
            return torch.as_tensor(normed, device=x.device, dtype=x.dtype)
        return normed

    def unnormalize(self, x):
        arr = _to_array(x)
        if arr is None:
            return None
        if self.clip and _uses_bounded_normalized_range(self.mode):
            arr = np.clip(arr, -1.0, 1.0)
        unnorm = _unnormalize_array(
            arr,
            mean=self.mean,
            std=self.std,
            min_val=self.min_val,
            max_val=self.max_val,
            q_low=self.q_low,
            q_high=self.q_high,
            mode=self.mode,
        )
        if self.mask is not None:
            unnorm = np.where(self.mask, unnorm, arr)
        if torch.is_tensor(x):
            return torch.as_tensor(unnorm, device=x.device, dtype=x.dtype)
        return unnorm


@dataclass
class RobotProcessor:
    """Normalizes and unnormalizes robot state/action features using dataset statistics."""

    action_normalizers: Dict[str, _FeatureNormalizer] = field(default_factory=dict)
    state_normalizers: Dict[str, _FeatureNormalizer] = field(default_factory=dict)
    repo_to_tag: Dict[str, str] = field(default_factory=dict)
    metadata_by_tag: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    action_stats_by_tag: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    state_stats_by_tag: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def normalize_action(self, action, repo_id: Optional[str]) -> Optional[object]:
        normalizer = self._get_normalizer(self.action_normalizers, repo_id)
        if normalizer is None:
            return action
        return normalizer.normalize(action)

    def normalize_state(self, state, repo_id: Optional[str]) -> Optional[object]:
        normalizer = self._get_normalizer(self.state_normalizers, repo_id)
        if normalizer is None:
            return state
        return normalizer.normalize(state)

    def unnormalize_action(self, action, repo_id: Optional[str]) -> Optional[object]:
        normalizer = self._get_normalizer(self.action_normalizers, repo_id)
        if normalizer is None:
            return action
        return normalizer.unnormalize(action)

    def unnormalize_state(self, state, repo_id: Optional[str]) -> Optional[object]:
        normalizer = self._get_normalizer(self.state_normalizers, repo_id)
        if normalizer is None:
            return state
        return normalizer.unnormalize(state)

    def _get_normalizer(
        self, mapping: Mapping[str, _FeatureNormalizer], repo_id: Optional[str]
    ) -> Optional[_FeatureNormalizer]:
        tag = self._resolve_tag(repo_id)
        return mapping.get(tag)

    def _get_feature_dim(
        self,
        normalizer_mapping: Mapping[str, _FeatureNormalizer],
        stats_mapping: Mapping[str, Mapping[str, Any]],
        repo_id: Optional[str],
    ) -> Optional[int]:
        tag = self._resolve_tag(repo_id)
        if tag is None:
            return None
        dim = _feature_dim_from_normalizer(normalizer_mapping.get(tag))
        if dim is not None:
            return dim
        return _feature_dim_from_stats(stats_mapping.get(tag))

    def _resolve_tag(self, repo_id: Optional[str]) -> Optional[str]:
        if repo_id is None:
            return None
        return self.repo_to_tag.get(repo_id, repo_id)

    def resolve_tag(self, repo_id: Optional[str]) -> Optional[str]:
        return self._resolve_tag(repo_id)

    def get_metadata(self, repo_id: Optional[str]) -> Dict[str, Any]:
        tag = self._resolve_tag(repo_id)
        if tag is None:
            return {}
        return dict(self.metadata_by_tag.get(tag, {}) or {})

    def _get_metadata_positive_int(self, repo_id: Optional[str], key: str) -> Optional[int]:
        metadata = self.get_metadata(repo_id)
        value = metadata.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Robot metadata for repo_id={repo_id!r} must define integer {key} >= 1."
            )
        value = int(value)
        if value < 1:
            raise ValueError(
                f"Robot metadata for repo_id={repo_id!r} must define integer {key} >= 1."
            )
        return value

    def get_action_dim(self, repo_id: Optional[str]) -> Optional[int]:
        return self._get_feature_dim(self.action_normalizers, self.action_stats_by_tag, repo_id)

    def get_state_dim(self, repo_id: Optional[str]) -> Optional[int]:
        return self._get_feature_dim(self.state_normalizers, self.state_stats_by_tag, repo_id)

    def get_action_horizon(self, repo_id: Optional[str]) -> Optional[int]:
        return self._get_metadata_positive_int(repo_id, "action_horizon")

    def get_n_action_steps(self, repo_id: Optional[str]) -> Optional[int]:
        return self._get_metadata_positive_int(repo_id, "n_action_steps")


@dataclass
class RobotProcessorConfig(BaseConfig):
    """Configuration container for robot normalization processing."""

    metadata_by_tag: Dict[str, Dict[str, Any]] = field(default_factory=dict, metadata={"allow_objects": True})
    norm_mode: str = "min_max"
    #: Saturate bounded modes at [-1, 1]. Absent from configs written before the
    #: switch existed, which therefore keep clipping exactly as they trained.
    norm_clip: bool = True

    @property
    def repo_to_tag(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for tag, metadata in self.metadata_by_tag.items():
            for repo_id in metadata.get("repo_ids", []) or []:
                mapping[str(repo_id)] = str(tag)
        return mapping

    @property
    def tag_metadata(self) -> Dict[str, Dict[str, Any]]:
        cleaned: Dict[str, Dict[str, Any]] = {}
        for tag, metadata in self.metadata_by_tag.items():
            cleaned[tag] = {
                str(key): _to_serializable(value)
                for key, value in metadata.items()
                if key not in {"state_stats", "action_stats"}
            }
        return cleaned

    @property
    def stats_by_tag(self) -> Dict[str, Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for tag, metadata in self.metadata_by_tag.items():
            tag_stats: Dict[str, Any] = {}
            if metadata.get("state_keys"):
                state_keys = _normalize_state_keys(metadata.get("state_keys"))
            elif metadata.get("state_key"):
                state_keys = _normalize_state_keys([metadata.get("state_key")])
            else:
                state_keys = _normalize_state_keys(None)
            state_stats_key = str(state_keys[0])
            action_key = str(metadata.get("action_key") or "action")
            state_stats = metadata.get("state_stats")
            action_stats = metadata.get("action_stats")
            if state_stats is not None:
                tag_stats[state_stats_key] = _to_serializable(state_stats)
            if action_stats is not None:
                tag_stats[action_key] = _to_serializable(action_stats)
            if tag_stats:
                merged[str(tag)] = tag_stats
        return merged

    @property
    def action_key(self) -> str:
        for metadata in self.metadata_by_tag.values():
            if metadata.get("action_key"):
                return str(metadata["action_key"])
        return "action"

    @property
    def state_keys(self) -> List[str]:
        for metadata in self.metadata_by_tag.values():
            if metadata.get("state_keys"):
                return _normalize_state_keys(metadata["state_keys"])
            if metadata.get("state_key"):
                return [str(metadata["state_key"])]
        return ["observation.state"]

    def build_processor(self) -> RobotProcessor:
        action_norms, state_norms = self._build_normalizers()
        return RobotProcessor(
            action_normalizers=action_norms,
            state_normalizers=state_norms,
            repo_to_tag=self.repo_to_tag,
            metadata_by_tag={
                str(tag): {
                    str(key): _to_serializable(value)
                    for key, value in metadata.items()
                    if key not in {"action_stats", "state_stats"}
                }
                for tag, metadata in self.metadata_by_tag.items()
            },
            action_stats_by_tag={
                str(tag): _to_serializable(metadata.get("action_stats"))
                for tag, metadata in self.metadata_by_tag.items()
                if metadata.get("action_stats") is not None
            },
            state_stats_by_tag={
                str(tag): _to_serializable(metadata.get("state_stats"))
                for tag, metadata in self.metadata_by_tag.items()
                if metadata.get("state_stats") is not None
            },
        )

    def _build_normalizers(self):
        action_norms: Dict[str, _FeatureNormalizer] = {}
        state_norms: Dict[str, _FeatureNormalizer] = {}
        for tag, metadata in self.metadata_by_tag.items():
            action_stats = metadata.get("action_stats")
            state_stats = metadata.get("state_stats")
            if action_stats is not None:
                try:
                    norm = _FeatureNormalizer.from_stats(
                        action_stats, mode=self.norm_mode, clip=self.norm_clip
                    )
                except ValueError as exc:
                    raise ValueError(f"Invalid action_stats for tag {tag!r}: {exc}") from exc
                if norm is not None:
                    action_norms[tag] = norm
            if state_stats is not None:
                try:
                    norm = _FeatureNormalizer.from_stats(
                        state_stats, mode=self.norm_mode, clip=self.norm_clip
                    )
                except ValueError as exc:
                    raise ValueError(f"Invalid state_stats for tag {tag!r}: {exc}") from exc
                if norm is not None:
                    state_norms[tag] = norm
        return action_norms, state_norms

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        if "norm_mode" not in config:
            if "action_norm_mode" in config and config.action_norm_mode is not None:
                config.norm_mode = config.action_norm_mode
            elif "state_norm_mode" in config and config.state_norm_mode is not None:
                config.norm_mode = config.state_norm_mode
        if "stats_by_tag" not in config and "stats_by_repo" in config:
            config.stats_by_tag = config.stats_by_repo
        if "metadata_by_tag" not in config:
            stats_by_tag = dict(config.get("stats_by_tag", {}) or {})
            tag_metadata = dict(config.get("tag_metadata", {}) or {})
            repo_to_tag = dict(config.get("repo_to_tag", {}) or {})
            default_action_key = str(config.get("action_key", "action"))
            default_state_keys = _normalize_state_keys(config.get("state_keys"))
            metadata_by_tag: Dict[str, Dict[str, Any]] = {}
            all_tags = set(stats_by_tag.keys()) | set(tag_metadata.keys()) | set(repo_to_tag.values())
            for tag in all_tags:
                tag = str(tag)
                legacy_entry = dict(stats_by_tag.get(tag, {}) or {})
                metadata = {
                    str(key): _to_serializable(value)
                    for key, value in dict(tag_metadata.get(tag, {}) or {}).items()
                }
                for key in (
                    "action_key",
                    "state_keys",
                    "camera_keys",
                    "camera_keys_alternative",
                    "setup_type",
                    "control_mode",
                    "repo_ids",
                ):
                    if key not in metadata and key in legacy_entry:
                        metadata[key] = _to_serializable(legacy_entry[key])
                action_key = str(metadata.get("action_key") or default_action_key)
                if "state_keys" in metadata:
                    state_key_list = _normalize_state_keys(metadata.get("state_keys"), default=default_state_keys)
                elif "state_key" in metadata:
                    state_key_list = _normalize_state_keys([metadata.get("state_key")], default=default_state_keys)
                else:
                    state_key_list = list(default_state_keys)
                state_stats_key = str(state_key_list[0])
                action_stats = legacy_entry.get("action_stats")
                if action_stats is None:
                    action_stats = legacy_entry.get(action_key)
                state_stats = legacy_entry.get("state_stats")
                if state_stats is None:
                    state_stats = legacy_entry.get(state_stats_key)
                action_stats = _to_serializable(action_stats)
                state_stats = _to_serializable(state_stats)
                if isinstance(action_stats, dict) and metadata.get("action_mask") is not None and "mask" not in action_stats:
                    action_stats["mask"] = _to_serializable(metadata["action_mask"])
                if isinstance(state_stats, dict) and metadata.get("state_mask") is not None and "mask" not in state_stats:
                    state_stats["mask"] = _to_serializable(metadata["state_mask"])
                metadata["action_key"] = action_key
                metadata["state_keys"] = state_key_list
                metadata.pop("state_key", None)
                if action_stats is not None:
                    metadata["action_stats"] = action_stats
                if state_stats is not None:
                    metadata["state_stats"] = state_stats
                metadata.pop("action_mask", None)
                metadata.pop("state_mask", None)
                repo_ids = sorted(str(repo_id) for repo_id, mapped_tag in repo_to_tag.items() if str(mapped_tag) == tag)
                if repo_ids:
                    metadata["repo_ids"] = repo_ids
                metadata_by_tag[tag] = metadata
            config.metadata_by_tag = metadata_by_tag
        if "action_norm_mode" in config:
            del config["action_norm_mode"]
        if "state_norm_mode" in config:
            del config["state_norm_mode"]
        if "stats_by_repo" in config:
            del config["stats_by_repo"]
        if "default_repo_id" in config:
            del config["default_repo_id"]
        for key in (
            "stats_by_tag",
            "repo_to_tag",
            "action_key",
            "state_keys",
            "tag_metadata",
            "data_formatter_add_setup_tokens",
            "data_formatter_add_control_tokens",
            "default_tag",
        ):
            if key in config:
                del config[key]
        return config

    @classmethod
    def from_stats(
        cls,
        stats_by_tag: Mapping[str, Mapping[str, Any]],
        *,
        action_key: str = "action",
        state_keys: Optional[Iterable[str]] = None,
        tag_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
        repo_to_tag: Optional[Mapping[str, str]] = None,
        norm_mode: str = "min_max",
        norm_clip: bool = True,
        data_formatter_add_setup_tokens: bool = False,
        data_formatter_add_control_tokens: bool = False,
    ) -> "RobotProcessorConfig":
        state_key_list = _normalize_state_keys(state_keys)
        default_state_keys = list(state_key_list)
        allowed_keys = set(state_key_list)
        allowed_keys.add(action_key)
        if tag_metadata:
            for metadata in tag_metadata.values():
                tag_action_key = metadata.get("action_key")
                if tag_action_key:
                    allowed_keys.add(str(tag_action_key))
                if metadata.get("state_keys"):
                    allowed_keys.update(_normalize_state_keys(metadata.get("state_keys")))
                elif metadata.get("state_key"):
                    allowed_keys.add(str(metadata.get("state_key")))

        repo_to_tag = dict(repo_to_tag or {})
        metadata_by_tag: Dict[str, Dict[str, Any]] = {}
        all_tags = set(str(tag) for tag in stats_by_tag.keys())
        all_tags.update(str(tag) for tag in (tag_metadata or {}).keys())
        all_tags.update(str(tag) for tag in repo_to_tag.values())
        for tag in all_tags:
            stats = dict(stats_by_tag.get(tag, {}) or {})
            tag = str(tag)
            metadata = {
                str(key): _to_serializable(value)
                for key, value in dict((tag_metadata or {}).get(tag, {}) or {}).items()
            }
            tag_stats: Dict[str, Any] = {}
            for key in allowed_keys:
                if key not in stats:
                    continue
                feature_stats = stats[key]
                if isinstance(feature_stats, Mapping):
                    tag_stats[key] = {k: _to_serializable(v) for k, v in feature_stats.items()}
                else:
                    tag_stats[key] = _to_serializable(feature_stats)
            resolved_action_key = str(metadata.get("action_key") or action_key)
            if metadata.get("state_keys"):
                resolved_state_keys = _normalize_state_keys(metadata.get("state_keys"), default=default_state_keys)
            elif metadata.get("state_key"):
                resolved_state_keys = _normalize_state_keys([metadata.get("state_key")], default=default_state_keys)
            else:
                resolved_state_keys = list(default_state_keys)
            resolved_state_stats_key = str(resolved_state_keys[0])
            metadata["action_key"] = resolved_action_key
            metadata["state_keys"] = resolved_state_keys
            metadata.pop("state_key", None)
            action_stats = tag_stats.get(resolved_action_key)
            state_stats = tag_stats.get(resolved_state_stats_key)
            if isinstance(action_stats, dict) and metadata.get("action_mask") is not None and "mask" not in action_stats:
                action_stats["mask"] = _to_serializable(metadata["action_mask"])
            if isinstance(state_stats, dict) and metadata.get("state_mask") is not None and "mask" not in state_stats:
                state_stats["mask"] = _to_serializable(metadata["state_mask"])
            if action_stats is not None:
                metadata["action_stats"] = action_stats
            if state_stats is not None:
                metadata["state_stats"] = state_stats
            metadata.pop("action_mask", None)
            metadata.pop("state_mask", None)
            repo_ids = sorted(str(repo_id) for repo_id, mapped_tag in repo_to_tag.items() if str(mapped_tag) == tag)
            if repo_ids:
                metadata["repo_ids"] = repo_ids
            metadata_by_tag[tag] = metadata

        config = cls(
            metadata_by_tag=metadata_by_tag,
            norm_mode=norm_mode,
            norm_clip=norm_clip,
        )
        config._build_normalizers()
        return config
