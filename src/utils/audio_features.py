from __future__ import annotations

from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd

DEFAULT_FEATURE_COLUMNS = [
    "length",
    "chroma_stft_mean", "chroma_stft_var",
    "rms_mean", "rms_var",
    "spectral_centroid_mean", "spectral_centroid_var",
    "spectral_bandwidth_mean", "spectral_bandwidth_var",
    "rolloff_mean", "rolloff_var",
    "zero_crossing_rate_mean", "zero_crossing_rate_var",
    "harmony_mean", "harmony_var",
    "perceptr_mean", "perceptr_var",
    "tempo",
    *[f"mfcc{i}_{stat}" for i in range(1, 21) for stat in ("mean", "var")],
]


def _mean_var(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.var(values))


def extract_features_from_wav(
    wav_path: str | Path,
    duration: float = 3.0,
    sr: int = 22050,
    feature_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Extract GTZAN-style features from the first `duration` seconds of a WAV file."""
    y, sr = librosa.load(str(wav_path), sr=sr, mono=True, duration=duration)
    if y.size == 0:
        raise ValueError("Audio file is empty or cannot be decoded.")

    row: dict[str, float] = {"length": float(len(y))}

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    row["chroma_stft_mean"], row["chroma_stft_var"] = _mean_var(chroma)

    rms = librosa.feature.rms(y=y)
    row["rms_mean"], row["rms_var"] = _mean_var(rms)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    row["spectral_centroid_mean"], row["spectral_centroid_var"] = _mean_var(centroid)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    row["spectral_bandwidth_mean"], row["spectral_bandwidth_var"] = _mean_var(bandwidth)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    row["rolloff_mean"], row["rolloff_var"] = _mean_var(rolloff)

    zcr = librosa.feature.zero_crossing_rate(y)
    row["zero_crossing_rate_mean"], row["zero_crossing_rate_var"] = _mean_var(zcr)

    harmony, perceptr = librosa.effects.hpss(y)
    row["harmony_mean"], row["harmony_var"] = _mean_var(harmony)
    row["perceptr_mean"], row["perceptr_var"] = _mean_var(perceptr)

    tempo = librosa.feature.tempo(y=y, sr=sr)
    row["tempo"] = float(tempo[0])

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        row[f"mfcc{i + 1}_mean"] = float(np.mean(mfcc[i]))
        row[f"mfcc{i + 1}_var"] = float(np.var(mfcc[i]))

    columns = list(feature_columns) if feature_columns is not None else DEFAULT_FEATURE_COLUMNS
    missing = [col for col in columns if col not in row]
    if missing:
        raise ValueError(f"Missing extracted features: {missing}")

    return pd.DataFrame([[row[col] for col in columns]], columns=columns)
