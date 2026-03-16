from typing import Callable, Dict, List, Optional

import librosa
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

ProgressCallback = Callable[[float, str], None]

# Feature columns used by GTZAN features_3_sec.csv (excluding filename and label).
FEATURE_COLUMNS: List[str] = [
    "length",
    "chroma_stft_mean",
    "chroma_stft_var",
    "rms_mean",
    "rms_var",
    "spectral_centroid_mean",
    "spectral_centroid_var",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_var",
    "rolloff_mean",
    "rolloff_var",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_var",
    "harmony_mean",
    "harmony_var",
    "perceptr_mean",
    "perceptr_var",
    "tempo",
    "mfcc1_mean",
    "mfcc1_var",
    "mfcc2_mean",
    "mfcc2_var",
    "mfcc3_mean",
    "mfcc3_var",
    "mfcc4_mean",
    "mfcc4_var",
    "mfcc5_mean",
    "mfcc5_var",
    "mfcc6_mean",
    "mfcc6_var",
    "mfcc7_mean",
    "mfcc7_var",
    "mfcc8_mean",
    "mfcc8_var",
    "mfcc9_mean",
    "mfcc9_var",
    "mfcc10_mean",
    "mfcc10_var",
    "mfcc11_mean",
    "mfcc11_var",
    "mfcc12_mean",
    "mfcc12_var",
    "mfcc13_mean",
    "mfcc13_var",
    "mfcc14_mean",
    "mfcc14_var",
    "mfcc15_mean",
    "mfcc15_var",
    "mfcc16_mean",
    "mfcc16_var",
    "mfcc17_mean",
    "mfcc17_var",
    "mfcc18_mean",
    "mfcc18_var",
    "mfcc19_mean",
    "mfcc19_var",
    "mfcc20_mean",
    "mfcc20_var",
]


def _safe_var(x: np.ndarray) -> float:
    return float(np.var(x)) if x.size else 0.0


def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else 0.0


def _emit_progress(progress_callback: Optional[ProgressCallback], progress: float, message: str) -> None:
    if progress_callback is not None:
        progress_callback(progress, message)


def extract_features_dict(
    audio_path: str,
    sr: int = 22050,
    duration: float = 3.0,
    progress_callback: Optional[ProgressCallback] = None,
    use_tqdm: bool = False,
) -> Dict[str, float]:
    """Extract 3-second audio features compatible with GTZAN features_3_sec.csv."""
    pbar = None
    total_steps = 6
    if use_tqdm and tqdm is not None:
        pbar = tqdm(total=total_steps, desc="Extracting audio features", unit="step")

    _emit_progress(progress_callback, 0.05, "Loading audio")
    y, sr = librosa.load(audio_path, sr=sr, mono=True, duration=duration)

    _emit_progress(progress_callback, 0.15, "Standardizing to 3 seconds")
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    if pbar is not None:
        pbar.update(1)

    _emit_progress(progress_callback, 0.35, "Computing spectral features")
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    if pbar is not None:
        pbar.update(1)

    _emit_progress(progress_callback, 0.55, "Separating harmonic/percussive components")
    harmony, percussive = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if pbar is not None:
        pbar.update(1)

    _emit_progress(progress_callback, 0.75, "Computing MFCCs")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    if pbar is not None:
        pbar.update(1)

    _emit_progress(progress_callback, 0.9, "Assembling feature vector")
    features: Dict[str, float] = {
        "length": float(len(y)),
        "chroma_stft_mean": _safe_mean(chroma_stft),
        "chroma_stft_var": _safe_var(chroma_stft),
        "rms_mean": _safe_mean(rms),
        "rms_var": _safe_var(rms),
        "spectral_centroid_mean": _safe_mean(spectral_centroid),
        "spectral_centroid_var": _safe_var(spectral_centroid),
        "spectral_bandwidth_mean": _safe_mean(spectral_bandwidth),
        "spectral_bandwidth_var": _safe_var(spectral_bandwidth),
        "rolloff_mean": _safe_mean(rolloff),
        "rolloff_var": _safe_var(rolloff),
        "zero_crossing_rate_mean": _safe_mean(zero_crossing_rate),
        "zero_crossing_rate_var": _safe_var(zero_crossing_rate),
        "harmony_mean": _safe_mean(harmony),
        "harmony_var": _safe_var(harmony),
        "perceptr_mean": _safe_mean(percussive),
        "perceptr_var": _safe_var(percussive),
        "tempo": float(tempo),
    }

    for i in range(20):
        coef = mfcc[i]
        features[f"mfcc{i + 1}_mean"] = _safe_mean(coef)
        features[f"mfcc{i + 1}_var"] = _safe_var(coef)

    if pbar is not None:
        pbar.update(2)
        pbar.close()

    _emit_progress(progress_callback, 1.0, "Feature extraction completed")

    return features


def extract_features_frame(
    audio_path: str,
    feature_columns: List[str],
    progress_callback: Optional[ProgressCallback] = None,
    use_tqdm: bool = False,
) -> pd.DataFrame:
    """Return one-row DataFrame ordered by saved training columns."""
    feats = extract_features_dict(
        audio_path=audio_path,
        progress_callback=progress_callback,
        use_tqdm=use_tqdm,
    )
    row = {col: float(feats.get(col, 0.0)) for col in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)
