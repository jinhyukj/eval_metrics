import re
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List


def _normalize_ext(video_ext: str) -> str:
    if not video_ext:
        return ".mp4"
    if not video_ext.startswith("."):
        return "." + video_ext
    return video_ext


def _strip_cfr25(stem: str) -> str:
    suffix = "_cfr25"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _delimited_match(stem: str, key: str) -> bool:
    pattern = r"(^|[_-]){}($|[_-])".format(re.escape(key))
    return re.search(pattern, stem) is not None


def _find_key_from_known(stem: str, known_keys: Iterable[str]) -> Optional[str]:
    candidates = [key for key in known_keys if key and _delimited_match(stem, key)]
    if not candidates:
        return None
    candidates.sort(key=lambda key: (-len(key), key))
    if len(candidates) > 1 and len(candidates[0]) == len(candidates[1]):
        return None
    return candidates[0]


def _normalize_name(value: str) -> str:
    stem = Path(value).stem
    return _strip_cfr25(stem)


def load_name_list(path: Path) -> List[str]:
    names: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        names.append(_normalize_name(line))
    return names


def video_key(path: Path, known_keys: Optional[Iterable[str]] = None) -> str:
    stem = Path(path).stem
    if known_keys:
        matched = _find_key_from_known(stem, known_keys)
        if matched:
            return matched
    return _strip_cfr25(stem)


def build_video_maps(
    real_dir: Path,
    fake_dir: Path,
    video_ext: str = ".mp4",
    name_list_path: Optional[Path] = None,
) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    ext = _normalize_ext(video_ext)
    real_paths = list(real_dir.rglob("*" + ext))
    if name_list_path:
        known_keys = load_name_list(name_list_path)
    else:
        known_keys = [_normalize_name(path.name) for path in real_paths]
    real_map = {video_key(path, known_keys): path for path in real_paths}
    fake_map = {video_key(path, known_keys): path for path in fake_dir.rglob("*" + ext)}
    return real_map, fake_map
