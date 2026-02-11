#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RMSD分析插件

本模块用于计算分子动力学轨迹的均方根偏差(RMSD)，支持对齐和非对齐两种计算方式。
主要功能:
1. 读取多种格式的轨迹文件（PDB、DCD、XTC、TRR、NC等）
2. 计算相对于参考帧的RMSD值
3. 支持Kabsch对齐算法进行结构叠合
4. 输出RMSD数据文件（.dat格式）
5. 生成RMSD可视化图片（.png格式）
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Union, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Optional: MDTraj for real trajectory analysis
try:
    import mdtraj as md
    MDTRAJ_AVAILABLE = True
except ImportError:
    MDTRAJ_AVAILABLE = False


# =============================================================================
# Path Utilities (PyInstaller compatible)
# =============================================================================

def resource_path(relative_path):
    """
    获取资源文件的绝对路径

    该函数用于处理打包后的可执行文件中的资源路径问题。
    当程序被打包为可执行文件（如使用 PyInstaller）时，原资源文件会被解压到临时目录。
    此时需要使用 sys._MEIPASS 路径来访问资源文件。

    Args:
        relative_path (str): 资源文件的相对路径

    Returns:
        str: 资源文件的绝对路径
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)




# =============================================================================
# Custom Exception
# =============================================================================

class AnalysisError(Exception):
    """Custom exception class for analysis errors."""
    pass


# =============================================================================
# JSON Utilities
# =============================================================================

def load_json(file_path: str) -> dict:
    """
    加载 JSON 文件并返回其内容

    Args:
        file_path (str): JSON 文件的路径

    Returns:
        dict: JSON 文件的内容

    Raises:
        FileNotFoundError: 当指定的文件不存在时抛出
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data


# =============================================================================
# RMSD Calculation Functions
# =============================================================================

def rmsd_no_fit(coords: np.ndarray, ref: np.ndarray, atom_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute RMSD per frame without optimal superposition (no-fit).

    Parameters
    ----------
    coords : np.ndarray
        Trajectory coordinates with shape (T, N, 3) where T is frames, N is atoms.
    ref : np.ndarray
        Reference coordinates with shape (1, N, 3) or (N, 3).
    atom_indices : np.ndarray, optional
        Optional atom indices to select before computing.

    Returns
    -------
    np.ndarray
        RMSD values in nm for each frame, shape (T,).
    """
    X = np.asarray(coords)
    R = np.asarray(ref)

    # Ensure ref has correct shape
    if R.ndim == 2:
        R = R[np.newaxis, :, :]

    # Apply atom selection if provided
    if atom_indices is not None:
        X = X[:, atom_indices, :]
        R = R[:, atom_indices, :]

    # No-fit RMSD = sqrt(mean(||x_i - y_i||^2)) over atoms
    diff = X - R  # (T, N, 3)
    msd = np.mean(np.sum(diff * diff, axis=2), axis=1)

    return np.sqrt(msd).astype(np.float64)


def rmsd_with_alignment(coords: np.ndarray, ref: np.ndarray, atom_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute RMSD with optimal Kabsch alignment.

    For simplicity, this implementation uses the Kabsch algorithm.
    In production, mdtraj.rmsd() is preferred for efficiency.

    Parameters
    ----------
    coords : np.ndarray
        Trajectory coordinates with shape (T, N, 3).
    ref : np.ndarray
        Reference coordinates with shape (1, N, 3) or (N, 3).
    atom_indices : np.ndarray, optional
        Atom indices for selection.

    Returns
    -------
    np.ndarray
        RMSD values in nm for each frame, shape (T,).
    """
    X = np.asarray(coords)
    R = np.asarray(ref)

    if R.ndim == 2:
        R = R[np.newaxis, :, :]

    if atom_indices is not None:
        X = X[:, atom_indices, :]
        R = R[:, atom_indices, :]

    T = X.shape[0]
    rmsd_values = np.zeros(T, dtype=np.float64)
    ref_centered = R[0] - R[0].mean(axis=0)

    for i in range(T):
        # Center coordinates
        x_centered = X[i] - X[i].mean(axis=0)

        # Kabsch alignment
        H = x_centered.T @ ref_centered
        U, _, Vt = np.linalg.svd(H)

        # Handle reflection case
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1, 1, d])
        R_matrix = Vt.T @ D @ U.T

        # Apply rotation
        x_aligned = x_centered @ R_matrix.T

        # Compute RMSD
        diff = x_aligned - ref_centered
        rmsd_values[i] = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))

    return rmsd_values


# =============================================================================
# Data Generation (Demo/Example Data)
# =============================================================================

def generate_sample_data(n_frames: int = 100, n_atoms: int = 50, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Generate sample trajectory data for demonstration purposes.

    This simulates a protein equilibration where RMSD increases initially
    then plateaus, mimicking realistic MD behavior.

    Parameters
    ----------
    n_frames : int
        Number of trajectory frames.
    n_atoms : int
        Number of atoms.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with 'coords' (trajectory) and 'ref' (reference frame).
    """
    np.random.seed(seed)

    # Generate reference structure (random "folded" state)
    ref = np.random.randn(1, n_atoms, 3).astype(np.float64) * 0.5  # nm scale

    # Generate trajectory with gradual drift + noise
    coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float64)

    for i in range(n_frames):
        # Simulate equilibration: rapid initial change, then plateau
        drift_factor = 0.1 * (1 - np.exp(-i / 20))  # Exponential approach
        noise = np.random.randn(n_atoms, 3) * 0.02  # Thermal fluctuations

        # Add systematic drift in one direction
        systematic_drift = np.array([drift_factor * 0.5, drift_factor * 0.3, 0])

        coords[i] = ref[0] + drift_factor * np.random.randn(n_atoms, 3) * 0.3 + noise + systematic_drift

    return {"coords": coords, "ref": ref}


# =============================================================================
# I/O Helpers
# =============================================================================

def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return default


def load_coords_from_file(
    input_file: str,
    topology_file: Optional[str],
    logger: logging.Logger
) -> np.ndarray:
    ext = os.path.splitext(input_file)[1].lower()

    if ext in {".npy"}:
        coords = np.load(input_file)
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise AnalysisError(".npy must have shape (T, N, 3)")
        return coords.astype(np.float64)

    if ext in {".npz"}:
        data = np.load(input_file)
        if "coords" not in data:
            raise AnalysisError(".npz must contain 'coords' array")
        coords = data["coords"]
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise AnalysisError("'coords' must have shape (T, N, 3)")
        return coords.astype(np.float64)

    if not MDTRAJ_AVAILABLE:
        raise AnalysisError("mdtraj is required to read trajectory files")

    if ext in {".dcd", ".xtc", ".trr", ".nc", ".netcdf"} and not topology_file:
        raise AnalysisError("topology_file is required for trajectory formats")

    logger.info("Loading trajectory with mdtraj: %s", input_file)
    # For PDB files, don't pass topology if it's empty
    if topology_file and topology_file.strip():
        traj = md.load(input_file, top=topology_file)
    else:
        traj = md.load(input_file)
    return traj.xyz.astype(np.float64)


# =============================================================================
# Main Analysis Class
# =============================================================================

class RMSDAnalysis:
    """
    RMSD Analysis class for computing Root-Mean-Square Deviation.

    Calculates RMSD of trajectory frames relative to a reference frame,
    with optional alignment (Kabsch superposition).

    Attributes
    ----------
    coords : np.ndarray
        Trajectory coordinates.
    reference_frame : int
        Index of the reference frame.
    atoms : str or None
        Atom selection string (for MDTraj compatibility).
    align : bool
        Whether to perform Kabsch alignment.
    data : np.ndarray
        Computed RMSD values.
    results : dict
        Analysis results dictionary.
    """

    def __init__(
            self,
            coords: np.ndarray,
            reference_frame: int = 0,
            atoms: Optional[str] = None,
            align: bool = True,
            output_dir: str = "output",
            logger: Optional[logging.Logger] = None
    ):
        """
        Initialize RMSD analysis.

        Parameters
        ----------
        coords : np.ndarray
            Trajectory coordinates with shape (T, N, 3).
        reference_frame : int
            Reference frame index (default 0). Negative indices allowed.
        atoms : str, optional
            Atom selection string (not used in demo mode).
        align : bool
            If True, perform Kabsch alignment before RMSD calculation.
        output_dir : str
            Output directory for results.
        logger : logging.Logger, optional
            Logger instance.
        """
        self.coords = np.asarray(coords)
        self.n_frames = self.coords.shape[0]
        self.n_atoms = self.coords.shape[1]

        # Handle negative indices
        if reference_frame < 0:
            reference_frame = self.n_frames + reference_frame

        if not (0 <= reference_frame < self.n_frames):
            raise AnalysisError(f"Invalid reference frame index: {reference_frame}")

        self.reference_frame = reference_frame
        self.atoms = atoms
        self.align = align

        self.outdir = output_dir
        os.makedirs(self.outdir, exist_ok=True)

        self.logger = logger or logging.getLogger("rmsd_analysis")

        self.data: Optional[np.ndarray] = None
        self.results: Dict[str, Any] = {}

        self.logger.info(
            "Initialized RMSD analysis: reference_frame=%d, align=%s, n_frames=%d, n_atoms=%d",
            self.reference_frame, self.align, self.n_frames, self.n_atoms
        )

    def run(self) -> Dict[str, np.ndarray]:
        """
        Compute RMSD for each frame relative to the reference frame.

        Returns
        -------
        dict
            {"rmsd": (N, 1) array of RMSD values in nm}
        """
        try:
            # Get reference frame
            ref = self.coords[self.reference_frame:self.reference_frame + 1]
            self.logger.debug("Reference frame %d loaded successfully", self.reference_frame)

            self.logger.info(
                "Starting RMSD calculation: ref=%d, align=%s, n_frames=%d, n_atoms=%d",
                self.reference_frame, self.align, self.n_frames, self.n_atoms
            )

            if self.align:
                self.logger.debug("Computing aligned RMSD (Kabsch)")
                rmsd_values = rmsd_with_alignment(self.coords, ref)
            else:
                self.logger.debug("Computing no-fit RMSD")
                rmsd_values = rmsd_no_fit(self.coords, ref)

            self.data = rmsd_values.reshape(-1, 1)
            self.results = {"rmsd": self.data}

            rmsd_range = (self.data.min(), self.data.max())
            self.logger.info(
                "RMSD analysis complete: range [%.4f, %.4f] nm, mean=%.4f nm",
                rmsd_range[0], rmsd_range[1], self.data.mean()
            )

            return self.results

        except Exception as e:
            self.logger.exception("RMSD analysis failed")
            raise AnalysisError(f"RMSD analysis failed: {e}")

    def save_data(
            self,
            filename: str = "rmsd",
            header: str = "rmsd_nm",
            fmt: str = "%.6f"
    ) -> str:
        """
        Save RMSD data to a .dat file.

        Parameters
        ----------
        filename : str
            Base filename (without extension).
        header : str
            Header line for the data file.
        fmt : str
            Numeric format string.

        Returns
        -------
        Path
            Path to the saved file.
        """
        if self.data is None:
            raise AnalysisError("No RMSD data available. Please run the analysis first.")

        if filename.lower().endswith(".dat"):
            filename = filename[:-4]
        path = os.path.join(self.outdir, f"{filename}.dat")

        np.savetxt(
            path,
            self.data,
            fmt=fmt,
            header=header,
            comments=""
        )

        self.logger.info("Data saved to %s", path)
        return path

    def plot(
            self,
            data: Optional[np.ndarray] = None,
            title: Optional[str] = None,
            xlabel: str = "Frame",
            ylabel: str = "RMSD (nm)",
            color: str = "#1f77b4",
            linestyle: str = "-",
            marker: str = "o",
            figsize: tuple = (10, 6),
            dpi: int = 300,
            grid_alpha: float = 0.3,
            filename: str = "rmsd.png"
    ) -> str:
        """
        Generate a plot of RMSD versus frame number.

        Parameters
        ----------
        data : np.ndarray, optional
            RMSD data to plot. If None, uses self.data.
        title : str, optional
            Plot title. If None, auto-generated.
        xlabel : str
            X-axis label.
        ylabel : str
            Y-axis label.
        color : str
            Line color.
        linestyle : str
            Line style.
        marker : str
            Marker style.
        figsize : tuple
            Figure size (width, height).
        dpi : int
            Figure resolution.
        grid_alpha : float
            Grid transparency.
        filename : str
            Filename for the saved plot.

        Returns
        -------
        Path
            Path to the saved plot.
        """
        if data is None:
            data = self.data
        if data is None:
            raise AnalysisError("No RMSD data available to plot. Please run the analysis first.")

        self.logger.debug("Generating RMSD plot")

        y = np.asarray(data).reshape(-1)
        x = np.arange(1, y.size + 1, dtype=int)

        if title is None:
            title = f"RMSD vs Frame (ref={self.reference_frame}, align={self.align})"

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot data
        ax.plot(x, y, color=color, linestyle=linestyle, marker=marker,
                markersize=3, linewidth=1.5, alpha=0.8)

        # Labels and title
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        # Grid
        ax.grid(True, alpha=grid_alpha, linestyle='--')

        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set axis to start from zero
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        # Add statistics annotation
        stats_text = f"Mean: {y.mean():.4f} nm\nMax: {y.max():.4f} nm\nMin: {y.min():.4f} nm"
        ax.text(
            0.98, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        fig.tight_layout()

        if filename.lower().endswith(".png"):
            plot_path = os.path.join(self.outdir, filename)
        else:
            plot_path = os.path.join(self.outdir, f"{filename}.png")
        fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        self.logger.info("Plot saved to %s", plot_path)
        return plot_path


# =============================================================================
# Main Entry Point
# =============================================================================

def main(params_file_path):
    """
    主函数：执行RMSD分析流程

    该函数是整个程序的入口点，负责：
    1. 从 JSON 文件加载参数（输入文件、输出目录、参考帧等）
    2. 创建输出目录（如果不存在）
    3. 配置日志系统，同时输出到文件和控制台
    4. 读取轨迹坐标数据（支持多种格式或生成示例数据）
    5. 初始化 RMSDAnalysis 类进行 RMSD 计算
    6. 使用 Kabsch 算法进行结构对齐（可选）
    7. 将 RMSD 数据保存为 .dat 文件
    8. 使用 matplotlib 绘制 RMSD 图并保存为 PNG 图片

    Args:
        params_file_path (str): 参数配置文件的路径，通常为 params.json
    """
    try:
        params = load_json(params_file_path)
        
        output_dir = params.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)

        log_file = os.path.join(output_dir, 'run.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8', mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger("rmsd_analysis")
        logger.info(f"config.json: {params}")

        input_file = params.get("input_file")
        topology_file = params.get("topology_file")

        coords: np.ndarray
        if input_file:
            if not os.path.isabs(input_file):
                input_file = resource_path(input_file)
            if topology_file and not os.path.isabs(topology_file):
                topology_file = resource_path(topology_file)
            coords = load_coords_from_file(input_file, topology_file, logger)
        else:
            logger.info("No input_file provided; using generated sample data")
            coords = generate_sample_data(
                n_frames=int(params.get("sample_frames", 100)),
                n_atoms=int(params.get("sample_atoms", 50))
            )["coords"]

        ref_frame = int(params.get("reference_frame", 0))
        align = parse_bool(params.get("align", True), default=True)

        analysis = RMSDAnalysis(
            coords=coords,
            reference_frame=ref_frame,
            align=align,
            output_dir=output_dir,
            logger=logger
        )

        results = analysis.run()

        data_filename = params.get("data_filename", "rmsd")
        data_header = params.get("data_header", "rmsd_nm")
        data_format = params.get("data_format", "%.6f")
        data_path = analysis.save_data(filename=data_filename, header=data_header, fmt=data_format)
        
        logger.info(f"RMSD数据已保存为 {data_path}")

        plot_kwargs = {
            "title": params.get("plot_title"),
            "xlabel": params.get("plot_xlabel", "Frame"),
            "ylabel": params.get("plot_ylabel", "RMSD (nm)"),
            "color": params.get("plot_color", "#1f77b4"),
            "linestyle": params.get("plot_linestyle", "-"),
            "marker": params.get("plot_marker", "o"),
            "figsize": tuple(params.get("plot_figsize", [10, 6])),
            "dpi": int(params.get("plot_dpi", 300)),
            "grid_alpha": float(params.get("plot_grid_alpha", 0.3)),
            "filename": params.get("plot_filename", "rmsd.png")
        }
        plot_path = analysis.plot(**plot_kwargs)
        
        logger.info(f"RMSD可视化已保存为 {plot_path}")
        logger.info("Analysis complete")
        
        return results

    except Exception as e:
        logging.exception(f"Error: {e}")
        raise


if __name__ == "__main__":
    """
    程序入口点

    当程序直接运行（而非被导入为模块）时，执行以下操作：
    1. 从命令行参数获取参数配置文件的路径
    2. 调用 main 函数开始处理

    使用方式：
        python plot.py params.json
    """
    params_file_path = sys.argv[1]
    main(params_file_path)
