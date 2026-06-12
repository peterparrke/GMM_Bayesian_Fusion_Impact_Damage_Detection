"""GMM-Bayesian fusion pipeline for interfacial debonding detection."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.mixture import GaussianMixture

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **_kwargs):
        return x


@dataclass
class PipelineConfig:
    data_root: Path = Path("data")
    output_root: Path = Path("article_python_outputs")
    datasets: tuple[str, ...] = ("pdd-1", "pdd-2", "pdd-3", "pdd-4")
    prefix: str = "pdd"

    row: int = 100000
    sample_rate: float = 1_000_000.0
    rows_per_point: int = 4
    grid_rows: int = 17
    grid_cols: int = 17
    hammer_row_offset: int = 1
    channel_suffixes: tuple[str, ...] = ("7", "8", "9", "10")
    channel_modalities: tuple[str, ...] = ("hammer", "acc", "highfre", "lowfre")
    value_column: str = "last"  # last, first, or flatten
    apply_hammer_normalization: bool = True
    feature_epsilon: float = 1e-12

    selected_features: tuple[str, ...] = (
        "ClearanceF",
        "ImpulseF",
        "KurtosisF",
        "ShapeF",
        "ZCR",
    )

    psd_data_len: int = 100000
    psd_nfft: int = 131072
    psd_win_len: int = 8192
    psd_bands: tuple[tuple[str, float, float, bool], ...] = (
        ("band1_0_2k5", 0.0, 2500.0, False),
        ("band2_2k5_10k", 2500.0, 10000.0, False),
        ("band3_10k_30k", 10000.0, 30000.0, True),
    )

    gmm_components: int = 2
    gmm_regularization: float = 1e-6
    gmm_replicates: int = 5
    gmm_max_iter: int = 2000
    gmm_tol: float = 1e-4
    gmm_random_state: int | None = 43
    log_epsilon: float = 1e-12
    constant_map_tol: float = 1e-12

    gmm_threshold: float = 0.5
    bayes_input_threshold: float = 0.5
    bayes_output_threshold: float = 0.5
    bayes_max_iter: int = 100
    bayes_epsilon: float = 1e-4

    gt_mat_file: Path | None = None
    max_workers_io: int = field(default_factory=lambda: min(8, (os.cpu_count() or 4) * 2))
    max_workers_cpu: int = field(default_factory=lambda: min(61, max(1, (os.cpu_count() or 4) - 1)))

    @property
    def psd_overlap(self) -> int:
        return self.psd_win_len // 2

    @property
    def recording_duration(self) -> float:
        return self.row / self.sample_rate

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.grid_rows, self.grid_cols

    @property
    def features_per_modality(self) -> int:
        return len(self.selected_features) + len(self.psd_bands)

    def dirs(self) -> dict[str, Path]:
        root = self.output_root
        return {
            "root": root,
            "feature_extr": root / "feature_extr",
            "matrix_csv": root / "feature_extr" / "matrix_17x17_csv",
            "psd": root / "psd",
            "psd_ratio": root / "psd" / "psd_band_energy_ratio_csv",
            "features": root / "features",
            "gmm": root / "GMM_results",
            "bayes": root / "bayes_results_2",
            "tables": root / "tables",
        }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_config(cfg: PipelineConfig) -> None:
    if cfg.row != cfg.psd_data_len:
        raise ValueError("Configured runs require row == psd_data_len.")
    if cfg.psd_win_len > cfg.psd_data_len:
        raise ValueError("psd_win_len cannot exceed psd_data_len.")
    if cfg.psd_nfft < cfg.psd_win_len:
        raise ValueError("psd_nfft cannot be smaller than psd_win_len.")
    if not 0 <= cfg.hammer_row_offset < cfg.rows_per_point:
        raise ValueError("hammer_row_offset must index one of the per-point signal rows.")
    if cfg.channel_suffixes and len(cfg.channel_suffixes) != cfg.rows_per_point:
        raise ValueError("channel_suffixes must contain exactly rows_per_point entries.")
    if len(cfg.channel_modalities) != cfg.rows_per_point:
        raise ValueError("channel_modalities must contain exactly rows_per_point entries.")
    valid_modalities = {"lowfre", "hammer", "acc", "highfre"}
    invalid = sorted(set(cfg.channel_modalities) - valid_modalities)
    if invalid:
        raise ValueError(f"Unsupported channel modalities: {invalid}")
    if "hammer" not in cfg.channel_modalities:
        raise ValueError("channel_modalities must include hammer for force normalization.")
    if not 0.0 < cfg.gmm_tol < 1.0:
        raise ValueError("gmm_tol must be between 0 and 1.")
    if not 0.0 < cfg.bayes_epsilon < 0.5:
        raise ValueError("bayes_epsilon must be between 0 and 0.5.")
    if not 0.0 <= cfg.bayes_input_threshold <= 1.0:
        raise ValueError("bayes_input_threshold must be in [0, 1].")
    if not 0.0 <= cfg.bayes_output_threshold <= 1.0:
        raise ValueError("bayes_output_threshold must be in [0, 1].")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^\w\-]+", "_", str(value).strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "feature"


def read_numeric_signal(path: Path, max_rows: int | None = None, value_column: str = "last") -> np.ndarray:
    arr = np.loadtxt(path, dtype=float, max_rows=max_rows)
    if arr.ndim == 1:
        signal = arr
    elif value_column == "first":
        signal = arr[:, 0]
    elif value_column == "last":
        signal = arr[:, -1]
    elif value_column == "flatten":
        signal = arr.reshape(-1, order="F")
    else:
        raise ValueError(f"Unsupported value_column={value_column!r}")
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"Signal contains NaN or Inf: {path}")
    return signal


def extract_id(filename: str) -> str:
    stem = Path(filename).stem
    match = re.search(r"(\d{4}#\d+)", stem)
    return match.group(1) if match else stem


def matlab_matrix_from_vector(vec: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Equivalent to MATLAB reshape(vec, n_rows, n_cols).'."""
    return np.asarray(vec).reshape((n_rows, n_cols), order="F").T


def normalize_by_maximum(matrix: np.ndarray) -> np.ndarray:
    """Normalize a non-negative feature map by its maximum."""
    matrix = np.asarray(matrix, dtype=np.float64)
    maximum = float(np.max(matrix))
    if maximum <= 0.0:
        return np.zeros_like(matrix)
    return matrix / maximum


def signal_point_and_suffix(path: Path) -> tuple[str, str]:
    match = re.search(r"(\d{4})#(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Cannot parse point/channel suffix from filename: {path.name}")
    return match.group(1), match.group(2)


def sorted_signal_files(folder: Path, cfg: PipelineConfig) -> list[Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() == ".txt"]
    if not cfg.channel_suffixes:
        return sorted(files, key=lambda p: p.name)

    suffix_order = {suffix.lstrip("#"): idx for idx, suffix in enumerate(cfg.channel_suffixes)}
    grouped: dict[str, dict[str, Path]] = {}
    for path in files:
        point, suffix = signal_point_and_suffix(path)
        if suffix in suffix_order:
            grouped.setdefault(point, {})[suffix] = path

    ordered: list[Path] = []
    for point in sorted(grouped):
        by_suffix = grouped[point]
        missing = [suffix for suffix in suffix_order if suffix not in by_suffix]
        if missing:
            raise ValueError(f"{folder.name} point {point} missing channel suffixes: {missing}")
        ordered.extend(by_suffix[suffix] for suffix in sorted(suffix_order, key=suffix_order.get))
    return ordered


def hammer_offset(cfg: PipelineConfig) -> int:
    return cfg.channel_modalities.index("hammer")


def build_hammer_normalization(
    cfg: PipelineConfig,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Return per-file p_max/p_i factors using the hammer signal of each four-file point."""
    groups: list[tuple[str, int, list[Path], float]] = []
    for dataset in cfg.datasets:
        folder = cfg.data_root / dataset
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")
        files = sorted_signal_files(folder, cfg)
        if len(files) % cfg.rows_per_point != 0:
            raise ValueError(
                f"{dataset}: {len(files)} signal files is not divisible by "
                f"rows_per_point={cfg.rows_per_point}"
            )
        expected_files = cfg.grid_rows * cfg.grid_cols * cfg.rows_per_point
        if len(files) != expected_files:
            raise ValueError(
                f"{dataset}: configured grid requires {expected_files} files "
                f"({cfg.grid_rows}x{cfg.grid_cols}x{cfg.rows_per_point}), got {len(files)}"
            )
        for point_index in range(len(files) // cfg.rows_per_point):
            start = point_index * cfg.rows_per_point
            point_files = files[start : start + cfg.rows_per_point]
            hammer_path = point_files[hammer_offset(cfg)]
            hammer = read_numeric_signal(hammer_path, max_rows=cfg.row, value_column=cfg.value_column)
            if hammer.size < cfg.row:
                raise ValueError(f"Signal length {hammer.size} < row={cfg.row}: {hammer_path}")
            peak = float(np.max(np.abs(hammer[: cfg.row])))
            if peak <= 0.0:
                raise ValueError(f"Hammer peak is zero: {hammer_path}")
            groups.append((dataset, point_index + 1, point_files, peak))

    global_peak = max(item[3] for item in groups)
    scale_by_path: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for dataset, point_index, point_files, peak in groups:
        scale = global_peak / peak if cfg.apply_hammer_normalization else 1.0
        for path in point_files:
            scale_by_path[str(path.resolve())] = scale
        rows.append(
            {
                "dataset": dataset,
                "point_index": point_index,
                "hammer_file": point_files[hammer_offset(cfg)].name,
                "hammer_peak_abs": peak,
                "global_hammer_peak_abs": global_peak,
                "scale_factor": scale,
            }
        )
    return scale_by_path, pd.DataFrame(rows)


def save_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.savetxt(path, np.asarray(matrix, dtype=np.float64), delimiter=",")


def read_matrix_csv(path: Path) -> np.ndarray:
    try:
        arr = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
    except ValueError:
        arr = pd.read_csv(path).to_numpy(dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"CSV is not a matrix: {path}")
    return arr


def save_image(matrix: np.ndarray, title: str, path: Path, is_probability: bool = True) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(matrix, aspect="equal", interpolation="nearest")
    if is_probability:
        im.set_clim(0.0, 1.0)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_one_txt(args: tuple[Path, int, str, float]) -> np.ndarray:
    path, row, value_column, scale = args
    signal = read_numeric_signal(path, max_rows=row, value_column=value_column)
    if signal.size < row:
        raise ValueError(f"Signal length {signal.size} < row={row}: {path}")
    return signal[:row] * scale



def compute_features_for_folder(
    folder: Path,
    cfg: PipelineConfig,
    scale_by_path: dict[str, float],
) -> pd.DataFrame:
    files = sorted_signal_files(folder, cfg)
    if not files:
        raise FileNotFoundError(f"No .txt files found in {folder}")

    signals: list[np.ndarray] = [None] * len(files)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers_io)) as pool:
        futures = {
            pool.submit(
                _load_one_txt,
                (path, cfg.row, cfg.value_column, scale_by_path.get(str(path.resolve()), 1.0)),
            ): idx
            for idx, path in enumerate(files)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Read {folder.name}"):
            signals[futures[fut]] = fut.result()

    signals_arr = np.column_stack(signals)
    n_samples, n_files = signals_arr.shape

    means = np.mean(signals_arr, axis=0)
    ptpa = np.ptp(signals_arr, axis=0)
    stds = np.std(signals_arr, axis=0)
    rms = np.sqrt(np.mean(signals_arr ** 2, axis=0))
    kurts = kurtosis(signals_arr, axis=0, fisher=False, bias=False)
    skews = skew(signals_arr, axis=0, bias=False)
    mean_abs = np.mean(np.abs(signals_arr), axis=0)
    max_abs = np.max(np.abs(signals_arr), axis=0)

    sqrt_abs_mean = np.mean(np.sqrt(np.abs(signals_arr)), axis=0)

    kurtosis_f = kurts
    shape_f = rms / (mean_abs + cfg.feature_epsilon)
    impulse_f = max_abs / (mean_abs + cfg.feature_epsilon)
    clearance_f = max_abs / (sqrt_abs_mean ** 2 + cfg.feature_epsilon)
    zcr = np.sum(signals_arr[:-1] * signals_arr[1:] < 0, axis=0) / max(n_samples - 1, 1)

    feature_dict = {
        "Mean": means,
        "PTPA": ptpa,
        "Std": stds,
        "RMS": rms,
        "Kurt": kurts,
        "Skew": skews,
        "KurtosisF": kurtosis_f,
        "ShapeF": shape_f,
        "ImpulseF": impulse_f,
        "ClearanceF": clearance_f,
        "ZCR": zcr,
    }
    df = pd.DataFrame(feature_dict, index=[p.name for p in files])
    df.index.name = "File"
    return df


def run_feature_extraction(cfg: PipelineConfig, scale_by_path: dict[str, float]) -> None:
    dirs = cfg.dirs()
    ensure_dir(dirs["feature_extr"])
    merged = []
    for dataset in cfg.datasets:
        folder = cfg.data_root / dataset
        out_dir = dirs["feature_extr"] / dataset
        ensure_dir(out_dir)
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")
        df = compute_features_for_folder(folder, cfg, scale_by_path)
        df.to_csv(out_dir / "full_features_numpy.csv", encoding="utf-8-sig")
        selected = df.loc[:, [c for c in cfg.selected_features if c in df.columns]]
        missing = sorted(set(cfg.selected_features) - set(selected.columns))
        if missing:
            raise ValueError(f"{dataset} missing selected features: {missing}")
        selected.to_csv(out_dir / f"{dataset}_selected_features.csv", encoding="utf-8-sig")
        df2 = df.copy()
        df2.insert(0, "Dataset", dataset)
        merged.append(df2.reset_index())
    if merged:
        pd.concat(merged, axis=0, ignore_index=True).to_csv(
            dirs["feature_extr"] / "all_features_merged.csv",
            index=False,
            encoding="utf-8-sig",
        )


def run_time_feature_matrices(cfg: PipelineConfig) -> tuple[int, int]:
    dirs = cfg.dirs()
    reset_dir(dirs["matrix_csv"])
    final_shape = None
    for dataset in cfg.datasets:
        csv_path = dirs["feature_extr"] / dataset / f"{dataset}_selected_features.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Selected feature CSV not found: {csv_path}")
        table = pd.read_csv(csv_path, index_col=0, encoding="utf-8-sig")
        data = table.to_numpy(dtype=np.float64)
        n_rows, n_features = data.shape
        if n_rows % cfg.rows_per_point != 0:
            raise ValueError(f"{dataset}: row count {n_rows} is not divisible by {cfg.rows_per_point}")
        n_points = n_rows // cfg.rows_per_point
        n_grid_r, n_grid_c = cfg.grid_shape
        if n_points != n_grid_r * n_grid_c:
            raise ValueError(
                f"{dataset}: configured grid requires {n_grid_r}x{n_grid_c}={n_grid_r*n_grid_c} "
                f"points, got {n_points}"
            )
        final_shape = cfg.grid_shape
        for offset, modality in enumerate(cfg.channel_modalities):
            block = data[offset :: cfg.rows_per_point, :]
            for feat_idx, feat_name in enumerate(table.columns[:n_features]):
                matrix = matlab_matrix_from_vector(block[:, feat_idx], n_grid_r, n_grid_c)
                matrix = normalize_by_maximum(matrix)
                out_name = f"{dataset}_{modality}_{sanitize_name(feat_name)}.csv"
                save_matrix_csv(dirs["matrix_csv"] / out_name, matrix)
    if final_shape is None:
        raise RuntimeError("No time feature matrices were generated.")
    return final_shape


def collect_id_paths(cfg: PipelineConfig) -> dict[str, dict[str, Path]]:
    id_paths: dict[str, dict[str, Path]] = {}
    for dataset in cfg.datasets:
        folder = cfg.data_root / dataset
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")
        for path in sorted_signal_files(folder, cfg):
            id_paths.setdefault(extract_id(path.name), {})[dataset] = path
    return id_paths


def compute_psd_energy(signal: np.ndarray, cfg: PipelineConfig) -> list[float]:
    if signal.size < cfg.psd_data_len:
        raise ValueError(f"Signal length {signal.size} < dataLen {cfg.psd_data_len}")
    x = signal[: cfg.psd_data_len]
    window = np.hanning(cfg.psd_win_len)
    freqs, pxx = welch(
        x,
        fs=cfg.sample_rate,
        window=window,
        nperseg=cfg.psd_win_len,
        noverlap=cfg.psd_overlap,
        nfft=cfg.psd_nfft,
        detrend=False,
        scaling="density",
        return_onesided=True,
        average="mean",
    )
    values = []
    for _name, low, high, include_high in cfg.psd_bands:
        if include_high:
            band = (freqs >= low) & (freqs <= high)
        else:
            band = (freqs >= low) & (freqs < high)
        values.append(float(np.trapz(pxx[band], freqs[band])))
    return values


def run_psd_energy(cfg: PipelineConfig, scale_by_path: dict[str, float]) -> pd.DataFrame:
    dirs = cfg.dirs()
    ensure_dir(dirs["psd"])
    id_paths = collect_id_paths(cfg)
    complete_ids = [key for key in id_paths if all(dataset in id_paths[key] for dataset in cfg.datasets)]
    if not complete_ids:
        raise RuntimeError("No IDs are present in all configured datasets.")

    rows = []
    for id_value in tqdm(complete_ids, desc="PSD energy"):
        record: list[object] = [id_value]
        for dataset in cfg.datasets:
            path = id_paths[id_value][dataset]
            signal = read_numeric_signal(path, value_column=cfg.value_column)
            signal = signal * scale_by_path.get(str(path.resolve()), 1.0)
            record.extend(compute_psd_energy(signal, cfg))
        rows.append(record)

    columns = ["filename"]
    for ch_idx in range(1, len(cfg.datasets) + 1):
        columns.extend(
            [
                f"ch{ch_idx}_band1_0_2k5",
                f"ch{ch_idx}_band2_2k5_10k",
                f"ch{ch_idx}_band3_10k_30k",
            ]
        )
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(dirs["psd"] / "psd_band_energy.csv", index=False, encoding="utf-8-sig")
    return df


def run_psd_ratio_matrices(cfg: PipelineConfig, psd_df: pd.DataFrame | None = None) -> tuple[int, int]:
    dirs = cfg.dirs()
    reset_dir(dirs["psd_ratio"])
    if psd_df is None:
        psd_df = pd.read_csv(dirs["psd"] / "psd_band_energy.csv", encoding="utf-8-sig")
    values = psd_df.iloc[:, 1:].to_numpy(dtype=np.float64)
    if values.shape[0] % cfg.rows_per_point != 0:
        raise ValueError(f"PSD table row count {values.shape[0]} is not divisible by {cfg.rows_per_point}")
    n_points = values.shape[0] // cfg.rows_per_point
    n_grid_r, n_grid_c = cfg.grid_shape
    if n_points != n_grid_r * n_grid_c:
        raise ValueError(
            f"PSD table: configured grid requires {n_grid_r}x{n_grid_c}={n_grid_r*n_grid_c} "
            f"points, got {n_points}"
        )
    n_bands = len(cfg.psd_bands)
    if values.shape[1] % n_bands != 0:
        raise ValueError(f"PSD column count {values.shape[1]} is not divisible by n_bands {n_bands}")
    n_channels = values.shape[1] // n_bands

    modality_blocks = [
        (modality, values[offset :: cfg.rows_per_point, :])
        for offset, modality in enumerate(cfg.channel_modalities)
    ]
    band_names = [item[0] for item in cfg.psd_bands]
    for modality, block in modality_blocks:
        for ch_idx in range(n_channels):
            dataset = cfg.datasets[ch_idx] if ch_idx < len(cfg.datasets) else f"{cfg.prefix}-{ch_idx + 1}"
            for band_idx, band_name in enumerate(band_names):
                col = ch_idx * n_bands + band_idx
                matrix = matlab_matrix_from_vector(block[:, col], n_grid_r, n_grid_c)
                matrix = normalize_by_maximum(matrix)
                out_name = f"{dataset}_{modality}_{band_name}.csv"
                save_matrix_csv(dirs["psd_ratio"] / out_name, matrix)
    return n_grid_r, n_grid_c


def build_features_folder(cfg: PipelineConfig) -> None:
    dirs = cfg.dirs()
    reset_dir(dirs["features"])
    for src_dir in (dirs["psd_ratio"], dirs["matrix_csv"]):
        for path in sorted(src_dir.glob("*.csv"), key=lambda p: p.name.lower()):
            shutil.copy2(path, dirs["features"] / path.name)


def prepare_gmm_values(matrix: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Feature map contains NaN or Inf.")
    if np.min(values) < -cfg.log_epsilon:
        raise ValueError("Maximum-normalized feature maps must be non-negative.")
    return np.log(np.maximum(values, cfg.log_epsilon))


def modality_from_name(name: str) -> str:
    low = name.lower()
    if "_acc_" in low:
        return "acc"
    if "_ham_" in low or "_hammer_" in low:
        return "ham"
    if "_highfre_" in low:
        return "high"
    if "_lowfre_" in low:
        return "low"
    return "unknown"


def feature_from_name(name: str) -> tuple[int, str]:
    low = name.lower()
    ordered = (
        ("clearancef", "ClearanceF"),
        ("impulsef", "ImpulseF"),
        ("kurtosisf", "KurtosisF"),
        ("shapef", "ShapeF"),
        ("_zcr", "ZCR"),
        ("band1_0_2k5", "PSD_0_2.5kHz"),
        ("band2_2k5_10k", "PSD_2.5_10kHz"),
        ("band3_10k_30k", "PSD_10_30kHz"),
    )
    for index, (token, label) in enumerate(ordered):
        if token in low:
            return index, label
    return 99, "unknown"


def gmm_file_order(paths: Sequence[Path]) -> list[Path]:
    order = {"acc": 0, "ham": 1, "high": 2, "low": 3, "unknown": 9}
    return sorted(
        paths,
        key=lambda p: (
            order[modality_from_name(p.name)],
            feature_from_name(p.name)[0],
            p.name.lower(),
        ),
    )


def fit_gmm_map(matrix: np.ndarray, cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    model_values = prepare_gmm_values(matrix, cfg)
    shape = model_values.shape
    vec = model_values.ravel(order="F").reshape(-1, 1)

    if np.std(vec) < cfg.constant_map_tol:
        prob = np.zeros(shape, dtype=np.float64)
        mask = np.zeros(shape, dtype=bool)
        info = {
            "status": "constant",
            "defect_component": -1,
            "defect_weight": 0.0,
            "defect_count": 0,
            "posterior_mean": 0.0,
            "posterior_max": 0.0,
        }
        return prob, mask, info

    gmm = GaussianMixture(
        n_components=cfg.gmm_components,
        covariance_type="diag",
        reg_covar=cfg.gmm_regularization,
        tol=cfg.gmm_tol,
        max_iter=cfg.gmm_max_iter,
        n_init=cfg.gmm_replicates,
        init_params="k-means++",
        random_state=cfg.gmm_random_state,
    )
    gmm.fit(vec)
    if not gmm.converged_:
        raise RuntimeError(f"GMM failed to converge within {cfg.gmm_max_iter} iterations.")
    labels = gmm.predict(vec)
    posterior = gmm.predict_proba(vec)
    weights = gmm.weights_.reshape(-1)
    defect_component = int(np.argmin(weights))
    defect_vec = posterior[:, defect_component]
    mask_vec = labels == defect_component
    prob = defect_vec.reshape(shape, order="F")
    mask = mask_vec.reshape(shape, order="F")
    means = gmm.means_.reshape(-1)
    variances = np.asarray(gmm.covariances_).reshape(cfg.gmm_components, -1)[:, 0]
    info = {
        "status": "ok",
        "defect_component": defect_component + 1,
        "defect_weight": float(weights[defect_component]),
        "defect_mean_logabs": float(means[defect_component]),
        "defect_std_logabs": float(np.sqrt(max(variances[defect_component], 0.0))),
        "healthy_component": int(np.argmax(weights)) + 1,
        "healthy_weight": float(np.max(weights)),
        "defect_count": int(mask.sum()),
        "posterior_mean": float(prob.mean()),
        "posterior_max": float(prob.max()),
        "converged": bool(gmm.converged_),
        "n_iter": int(gmm.n_iter_),
        "lower_bound": float(gmm.lower_bound_),
    }
    for idx in range(cfg.gmm_components):
        info[f"component{idx + 1}_weight"] = float(weights[idx])
        info[f"component{idx + 1}_mean_logabs"] = float(means[idx])
        info[f"component{idx + 1}_std_logabs"] = float(np.sqrt(max(variances[idx], 0.0)))
    return prob, mask, info


def run_gmm(cfg: PipelineConfig) -> dict[str, np.ndarray]:
    dirs = cfg.dirs()
    reset_dir(dirs["gmm"])
    feature_files = list(dirs["features"].glob("*.csv"))
    if not feature_files:
        raise RuntimeError(f"No feature CSVs found in {dirs['features']}")

    prob_by_dataset: dict[str, np.ndarray] = {}
    info_rows = []
    for dataset in cfg.datasets:
        paths = [p for p in feature_files if p.name.startswith(f"{dataset}_")]
        paths = gmm_file_order(paths)
        if not paths:
            raise FileNotFoundError(f"No feature maps found for {dataset}")
        expected_maps = 4 * cfg.features_per_modality
        if len(paths) != expected_maps:
            raise ValueError(f"{dataset}: expected {expected_maps} feature maps, got {len(paths)}")
        for modality_index, modality in enumerate(("acc", "ham", "high", "low")):
            block = paths[
                modality_index * cfg.features_per_modality :
                (modality_index + 1) * cfg.features_per_modality
            ]
            labels = [feature_from_name(path.name)[1] for path in block]
            expected_labels = [
                "ClearanceF",
                "ImpulseF",
                "KurtosisF",
                "ShapeF",
                "ZCR",
                "PSD_0_2.5kHz",
                "PSD_2.5_10kHz",
                "PSD_10_30kHz",
            ]
            if [modality_from_name(path.name) for path in block] != [modality] * cfg.features_per_modality:
                raise ValueError(f"{dataset}: invalid modality order for {modality}: {[p.name for p in block]}")
            if labels != expected_labels:
                raise ValueError(f"{dataset}: invalid feature order for {modality}: {labels}")
        matrices = [read_matrix_csv(path) for path in paths]
        base_shape = matrices[0].shape
        mask_all = np.zeros((*base_shape, len(paths)), dtype=bool)
        prob_all = np.zeros((*base_shape, len(paths)), dtype=np.float64)

        file_order_rows = []
        for idx, (path, matrix) in enumerate(tqdm(list(zip(paths, matrices)), desc=f"GMM {dataset}"), start=1):
            if matrix.shape != base_shape:
                raise ValueError(f"{path}: shape {matrix.shape} != expected {base_shape}")
            prob, mask, info = fit_gmm_map(matrix, cfg)
            prob_all[:, :, idx - 1] = prob
            mask_all[:, :, idx - 1] = mask
            modality = modality_from_name(path.name)
            feature_no, feature_label = feature_from_name(path.name)
            file_order_rows.append(
                {
                    "index": idx,
                    "file": path.name,
                    "modality": modality,
                    "feature_no": feature_no + 1,
                    "feature": feature_label,
                }
            )
            info_rows.append(
                {
                    "dataset": dataset,
                    "index": idx,
                    "file": path.name,
                    "modality": modality,
                    "feature_no": feature_no + 1,
                    "feature": feature_label,
                    **info,
                }
            )

        pd.DataFrame(file_order_rows).to_csv(dirs["gmm"] / f"GMM_File_Order_{dataset}.csv", index=False)
        sio.savemat(dirs["gmm"] / f"mask_all_{dataset}.mat", {"mask_all": mask_all})
        sio.savemat(dirs["gmm"] / f"prob_all_{dataset}.mat", {"prob_all": prob_all})
        np.save(dirs["gmm"] / f"mask_all_{dataset}.npy", mask_all)
        np.save(dirs["gmm"] / f"prob_all_{dataset}.npy", prob_all)

        fused_prob = np.mean(prob_all, axis=2)
        fused_mask = fused_prob >= cfg.gmm_threshold
        save_matrix_csv(dirs["gmm"] / f"fused_prob_{dataset}.csv", fused_prob)
        save_matrix_csv(dirs["gmm"] / f"fused_mask_{dataset}.csv", fused_mask.astype(float))
        save_image(fused_prob, f"{dataset}: GMM fused defect probability", dirs["gmm"] / f"fused_prob_{dataset}.png")
        save_image(
            fused_mask.astype(float),
            f"{dataset}: GMM fused defect mask",
            dirs["gmm"] / f"fused_mask_{dataset}.png",
            is_probability=False,
        )
        prob_by_dataset[dataset] = prob_all

    pd.DataFrame(info_rows).to_csv(dirs["gmm"] / "GMM_Feature_Info.csv", index=False, encoding="utf-8-sig")
    return prob_by_dataset


def bayes_binary_bernoulli_fusion(
    prob_all: np.ndarray,
    input_threshold: float = 0.5,
    output_threshold: float = 0.5,
    max_iter: int = 100,
    epsilon: float = 1e-4,
):
    """Binary Bernoulli EM fusion."""
    prob_all = np.asarray(prob_all, dtype=np.float64)
    if prob_all.ndim != 3:
        raise ValueError(f"prob_all must be H x W x N, got shape {prob_all.shape}")
    if not np.all(np.isfinite(prob_all)):
        raise ValueError("prob_all contains NaN or Inf.")
    n_rows, n_cols, n_sensor = prob_all.shape
    n_pix = n_rows * n_cols
    indicators = (prob_all.reshape(n_pix, n_sensor).T >= input_threshold).astype(np.float64)

    # Pixel-wise prior from the mean of binary channel indicators.
    prior = np.mean(indicators, axis=0)
    post = prior.copy()

    # Data-driven initialization of alpha/beta from the initial prior.
    den_a = max(float(np.sum(post)), epsilon)
    den_b = max(float(np.sum(1.0 - post)), epsilon)
    alpha = np.sum(indicators * post[None, :], axis=1) / den_a
    beta = np.sum(indicators * (1.0 - post)[None, :], axis=1) / den_b
    alpha = np.clip(alpha, epsilon, 1.0 - epsilon)
    beta = np.clip(beta, epsilon, 1.0 - epsilon)

    iter_loss: list[float] = []
    for _it in range(max_iter):
        log_l1 = np.sum(
            indicators * np.log(alpha[:, None])
            + (1.0 - indicators) * np.log(1.0 - alpha[:, None]),
            axis=0,
        )
        log_l0 = np.sum(
            indicators * np.log(beta[:, None])
            + (1.0 - indicators) * np.log(1.0 - beta[:, None]),
            axis=0,
        )
        log_prior_1 = np.full(n_pix, -np.inf, dtype=np.float64)
        log_prior_0 = np.full(n_pix, -np.inf, dtype=np.float64)
        has_prior_1 = prior > 0.0
        has_prior_0 = prior < 1.0
        log_prior_1[has_prior_1] = np.log(prior[has_prior_1])
        log_prior_0[has_prior_0] = np.log1p(-prior[has_prior_0])
        log_num_1 = log_l1 + log_prior_1
        log_num_0 = log_l0 + log_prior_0
        log_denominator = np.logaddexp(log_num_1, log_num_0)
        post_new = np.exp(log_num_1 - log_denominator)

        den_a = max(float(np.sum(post_new)), epsilon)
        den_b = max(float(np.sum(1.0 - post_new)), epsilon)
        alpha_new = np.sum(indicators * post_new[None, :], axis=1) / den_a
        beta_new = np.sum(indicators * (1.0 - post_new)[None, :], axis=1) / den_b
        alpha_new = np.clip(alpha_new, epsilon, 1.0 - epsilon)
        beta_new = np.clip(beta_new, epsilon, 1.0 - epsilon)

        loss = float(np.max(np.abs(post_new - post)))
        iter_loss.append(loss)
        post = post_new
        alpha = alpha_new
        beta = beta_new
        if loss < epsilon:
            break

    post_map = post.reshape(n_rows, n_cols)
    mask = post_map >= output_threshold
    reliab = np.column_stack([alpha, beta])
    return post_map, mask, reliab, np.asarray(iter_loss, dtype=np.float64)


def load_ground_truth(path: Path | None) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    mat = sio.loadmat(path)
    if "A" not in mat:
        raise ValueError(f"{path} does not contain variable A")
    return np.asarray(mat["A"], dtype=np.float64)


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float | int]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)
    return {"TP": tp, "FP": fp, "FN": fn, "recall": recall, "precision": precision, "F1": f1}


def save_iter_loss(out_dir: Path, tag: str, iter_loss: np.ndarray) -> None:
    if iter_loss.size == 0:
        iter_loss = np.array([0.0])
    np.savetxt(
        out_dir / f"iter_loss_{tag}.txt",
        np.column_stack([np.arange(1, iter_loss.size + 1), iter_loss]),
        fmt=["%d", "%.8g"],
        header="Iteration MaxDeltaPosterior",
    )
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(np.arange(1, iter_loss.size + 1), iter_loss, "-o", linewidth=1.4)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max posterior change")
    ax.set_title(f"Bayesian EM convergence - {tag}")
    fig.tight_layout()
    fig.savefig(out_dir / f"iter_loss_{tag}.png", dpi=160)
    plt.close(fig)


def run_bayes_fusion(cfg: PipelineConfig, prob_by_dataset: dict[str, np.ndarray] | None = None) -> None:
    dirs = cfg.dirs()
    reset_dir(dirs["bayes"])
    if prob_by_dataset is None:
        prob_by_dataset = {}
        for dataset in cfg.datasets:
            mat_path = dirs["gmm"] / f"prob_all_{dataset}.mat"
            if not mat_path.is_file():
                raise FileNotFoundError(f"GMM probability file not found: {mat_path}")
            prob_by_dataset[dataset] = np.asarray(sio.loadmat(mat_path)["prob_all"], dtype=np.float64)

    gt = load_ground_truth(cfg.gt_mat_file)
    per = cfg.features_per_modality
    expected_maps = 4 * per
    all_metrics = []
    for dataset, prob_all in prob_by_dataset.items():
        if prob_all.shape[2] != expected_maps:
            raise ValueError(
                f"{dataset}: expected {expected_maps} feature maps "
                f"(4 x {per}), got {prob_all.shape[2]}"
            )
        acc_data = prob_all[:, :, 0:per]
        ham_data = prob_all[:, :, per : 2 * per]
        high_data = prob_all[:, :, 2 * per : 3 * per]
        low_data = prob_all[:, :, 3 * per : 4 * per]
        groups = {
            "acc": acc_data,
            "ham": ham_data,
            "high": high_data,
            "low": low_data,
            "ahl": np.concatenate([acc_data, high_data, low_data], axis=2),
            "all4": np.concatenate([acc_data, ham_data, high_data, low_data], axis=2),
        }

        clean_name = dataset.replace("-", "")
        out_dir = dirs["bayes"] / clean_name
        ensure_dir(out_dir)
        save_payload: dict[str, object] = {
            "input_threshold": cfg.bayes_input_threshold,
            "output_threshold": cfg.bayes_output_threshold,
            "epsilon": cfg.bayes_epsilon,
            "dataset": dataset,
        }
        metrics_for_dataset = []
        gt_bin = None
        if gt is not None:
            if gt.shape != prob_all.shape[:2]:
                raise ValueError(
                    f"Ground truth shape {gt.shape} != configured grid {prob_all.shape[:2]}"
                )
            gt_bin = gt > 0.5

        for tag, data in groups.items():
            post, mask, reliab, iter_loss = bayes_binary_bernoulli_fusion(
                data,
                input_threshold=cfg.bayes_input_threshold,
                output_threshold=cfg.bayes_output_threshold,
                max_iter=cfg.bayes_max_iter,
                epsilon=cfg.bayes_epsilon,
            )
            save_payload[f"f_prob_{tag}"] = post
            save_payload[f"f_mask_{tag}"] = mask
            save_payload[f"rel_{tag}"] = reliab
            save_payload[f"it_{tag}"] = iter_loss
            save_matrix_csv(out_dir / f"prob_{tag}.csv", post)
            save_matrix_csv(out_dir / f"mask_{tag}.csv", mask.astype(float))
            pd.DataFrame(reliab, columns=["Alpha_TPR", "Beta_FPR"]).to_csv(out_dir / f"rel_{tag}.csv", index=False)
            save_image(post, f"{dataset} - {tag} fused posterior", out_dir / f"prob_{tag}.png")
            save_image(mask.astype(float), f"{dataset} - {tag} fused mask", out_dir / f"mask_{tag}.png", is_probability=False)
            save_iter_loss(out_dir, tag, iter_loss)
            if gt_bin is not None:
                metrics = compute_metrics(mask, gt_bin)
                row = {"dataset": dataset, "group": tag, **metrics}
                metrics_for_dataset.append(row)
                all_metrics.append(row)

        sio.savemat(out_dir / f"bayes_fused_all_{clean_name}.mat", save_payload)
        if metrics_for_dataset:
            metrics_df = pd.DataFrame(metrics_for_dataset)
            metrics_df.to_csv(out_dir / f"metrics_{clean_name}.csv", index=False)
            with open(out_dir / f"metrics_{clean_name}.txt", "w", encoding="utf-8") as handle:
                for row in metrics_for_dataset:
                    handle.write(
                        f"{row['group']:<4}: Recall = {row['recall']:.4f}, "
                        f"Precision = {row['precision']:.4f}, F1 = {row['F1']:.4f} "
                        f"(TP={row['TP']}, FP={row['FP']}, FN={row['FN']})\n"
                    )

    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(dirs["bayes"] / "all_metrics.csv", index=False, encoding="utf-8-sig")


def write_run_summary(cfg: PipelineConfig, elapsed: float, grid_shape: tuple[int, int]) -> None:
    dirs = cfg.dirs()
    ensure_dir(dirs["tables"])
    summary = {
        "data_root": str(cfg.data_root),
        "output_root": str(cfg.output_root),
        "datasets": ",".join(cfg.datasets),
        "grid_shape": f"{grid_shape[0]}x{grid_shape[1]}",
        "elapsed_seconds": elapsed,
    }
    pd.DataFrame([summary]).to_csv(dirs["tables"] / "run_summary.csv", index=False, encoding="utf-8-sig")
    with open(dirs["tables"] / "run_summary.txt", "w", encoding="utf-8") as handle:
        for key, value in summary.items():
            handle.write(f"{key}={value}\n")


def run_pipeline(cfg: PipelineConfig) -> None:
    t0 = time.time()
    validate_config(cfg)
    ensure_dir(cfg.output_root)
    dirs = cfg.dirs()
    print(f"[Output] {cfg.output_root.resolve()}")
    print("[0/6] Hammer-reference normalization")
    scale_by_path, normalization_table = build_hammer_normalization(cfg)
    ensure_dir(dirs["tables"])
    normalization_table.to_csv(
        dirs["tables"] / "hammer_normalization_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("[1/6] Time-domain feature extraction")
    run_feature_extraction(cfg, scale_by_path)
    print("[2/6] Time-domain feature matrices")
    matrix_shape = run_time_feature_matrices(cfg)
    print("[3/6] PSD energy and PSD ratio matrices")
    psd_df = run_psd_energy(cfg, scale_by_path)
    psd_shape = run_psd_ratio_matrices(cfg, psd_df)
    if matrix_shape != psd_shape:
        raise ValueError(f"Time feature grid {matrix_shape} != PSD grid {psd_shape}")
    print("[4/6] Build combined feature folder")
    build_features_folder(cfg)
    print("[5/6] Two-component minority GMM")
    prob_by_dataset = run_gmm(cfg)
    print("[6/6] Bayesian fusion")
    run_bayes_fusion(cfg, prob_by_dataset)
    elapsed = time.time() - t0
    write_run_summary(cfg, elapsed, matrix_shape)
    print(f"[Done] grid={matrix_shape[0]}x{matrix_shape[1]}, maps/dataset={4 * cfg.features_per_modality}")
    print(f"[Done] Results -> {cfg.output_root.resolve()}")
    print(f"[Done] Runtime {elapsed:.1f}s")


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="GMM + Bayesian fusion pipeline for debonding detection.")
    parser.add_argument("--data-root", default="data", help="Root folder containing dataset subfolders.")
    parser.add_argument("--output-root", default="article_python_outputs", help="Output root for all generated files.")
    parser.add_argument("--datasets", nargs="+", default=["pdd-1", "pdd-2", "pdd-3", "pdd-4"])
    parser.add_argument("--prefix", default="pdd")
    parser.add_argument(
        "--channel-suffixes",
        nargs="*",
        default=["7", "8", "9", "10"],
        help="Point-channel suffixes to keep, in per-point order.",
    )
    parser.add_argument(
        "--channel-modalities",
        nargs="*",
        default=["hammer", "acc", "highfre", "lowfre"],
        help="Modality names matching --channel-suffixes.",
    )
    parser.add_argument("--row", type=int, default=100000)
    parser.add_argument("--sample-rate", type=float, default=1_000_000.0)
    parser.add_argument("--value-column", choices=["last", "first", "flatten"], default="last")
    parser.add_argument("--psd-data-len", type=int, default=100000)
    parser.add_argument("--psd-nfft", type=int, default=131072)
    parser.add_argument("--psd-win-len", type=int, default=8192)
    parser.add_argument("--gmm-tol", type=float, default=1e-4)
    parser.add_argument("--gmm-random-state", default="43", help="Integer seed, or 'none' for an unfixed run.")
    parser.add_argument("--bayes-input-threshold", type=float, default=0.5)
    parser.add_argument("--bayes-output-threshold", type=float, default=0.5)
    parser.add_argument("--bayes-max-iter", type=int, default=100)
    parser.add_argument("--bayes-epsilon", type=float, default=1e-4)
    parser.add_argument("--gt-mat", default="none", help="Optional MAT file containing GT variable A; use 'none' to skip metrics.")
    parser.add_argument("--max-workers-io", type=int, default=None)
    parser.add_argument("--max-workers-cpu", type=int, default=None)
    args = parser.parse_args()

    random_state = None if str(args.gmm_random_state).lower() == "none" else int(args.gmm_random_state)
    gt_mat = None if str(args.gt_mat).lower() == "none" else Path(args.gt_mat)
    cfg = PipelineConfig(
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        datasets=tuple(args.datasets),
        prefix=args.prefix,
        channel_suffixes=tuple(args.channel_suffixes),
        channel_modalities=tuple(args.channel_modalities) if args.channel_modalities else ("hammer", "acc", "highfre", "lowfre"),
        row=args.row,
        sample_rate=args.sample_rate,
        value_column=args.value_column,
        psd_data_len=args.psd_data_len,
        psd_nfft=args.psd_nfft,
        psd_win_len=args.psd_win_len,
        gmm_tol=args.gmm_tol,
        gmm_random_state=random_state,
        bayes_input_threshold=args.bayes_input_threshold,
        bayes_output_threshold=args.bayes_output_threshold,
        bayes_max_iter=args.bayes_max_iter,
        bayes_epsilon=args.bayes_epsilon,
        gt_mat_file=gt_mat,
    )
    if args.max_workers_io is not None:
        cfg.max_workers_io = args.max_workers_io
    if args.max_workers_cpu is not None:
        cfg.max_workers_cpu = args.max_workers_cpu
    return cfg


if __name__ == "__main__":
    run_pipeline(parse_args())
