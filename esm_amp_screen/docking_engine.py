"""
High-Throughput Parallel Docking Engine.
Runs AutoDock Vina in parallel across all available CPU threads with progress tracking.
"""

import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from tqdm import tqdm

from esm_amp_screen.utils import logger, ensure_dir


def parse_vina_scores(output_text: str, docked_pdbqt: Optional[Path] = None) -> List[float]:
    """
    Parses binding affinities (kcal/mol) from Vina stdout or REMARK lines in docked PDBQT.
    Supports standard Vina table, local_only energy breakdown, and PDBQT remarks.
    """
    scores = []
    # 1. From stdout table: "   1      -7.4      0.000      0.000"
    for line in output_text.splitlines():
        m = re.match(r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+", line)
        if m:
            scores.append(float(m.group(2)))

    # 2. From stdout if local_only was used (prefer intermolecular interaction energy)
    if not scores:
        for line in output_text.splitlines():
            m = re.search(r"\(1\)\s+Final Intermolecular Energy\s*:\s*(-?\d+(?:\.\d+)?)", line)
            if m:
                scores.append(float(m.group(1)))
                break
        if not scores:
            for line in output_text.splitlines():
                m = re.search(r"Estimated Free Energy of Binding\s*:\s*(-?\d+(?:\.\d+)?)", line)
                if m:
                    scores.append(float(m.group(1)))
                    break

    # 3. From output PDBQT if stdout had no scores
    if not scores and docked_pdbqt and docked_pdbqt.exists():
        content = docked_pdbqt.read_text(encoding="utf-8", errors="ignore")
        for line in content.splitlines():
            m = re.search(r"REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)", line)
            if m:
                scores.append(float(m.group(1)))

    return scores


def dock_single_peptide(
    pep_id: str,
    ligand_pdbqt: Path,
    receptor_pdbqt: Path,
    grid: Optional[Dict[str, float]],
    out_dir: Path,
    vina_bin: str = "vina",
    exhaustiveness: int = 8,
    num_modes: int = 9,
    energy_range: float = 4.0,
    cpu_per_task: int = 1,
    local_only: bool = False,
    autobox: bool = False,
    timeout: int = 900
) -> Dict[str, Any]:
    """
    Worker function to dock a single peptide against the target receptor.
    Supports global grid docking or pocket-guided local induced-fit refinement.
    """
    ligand_pdbqt = Path(ligand_pdbqt)
    receptor_pdbqt = Path(receptor_pdbqt)
    out_dir = Path(out_dir)

    docked_out = out_dir / f"{pep_id}_docked.pdbqt"
    log_out = out_dir / f"{pep_id}_vina.log"

    cmd = [
        vina_bin,
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--out", str(docked_out),
        "--cpu", str(cpu_per_task)
    ]

    if local_only:
        cmd.append("--local_only")
        if autobox or not grid:
            cmd.append("--autobox")
        else:
            cmd.extend([
                "--center_x", str(grid["center_x"]),
                "--center_y", str(grid["center_y"]),
                "--center_z", str(grid["center_z"]),
                "--size_x", str(grid["size_x"]),
                "--size_y", str(grid["size_y"]),
                "--size_z", str(grid["size_z"])
            ])
    else:
        if not grid:
            raise ValueError(f"Grid coordinates required for global docking of {pep_id}")
        cmd.extend([
            "--center_x", str(grid["center_x"]),
            "--center_y", str(grid["center_y"]),
            "--center_z", str(grid["center_z"]),
            "--size_x", str(grid["size_x"]),
            "--size_y", str(grid["size_y"]),
            "--size_z", str(grid["size_z"]),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(num_modes),
            "--energy_range", str(energy_range)
        ])

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        log_out.write_text(res.stdout + "\n" + res.stderr, encoding="utf-8")

        scores = parse_vina_scores(res.stdout, docked_out)
        best_affinity = scores[0] if scores else 0.0
        success = res.returncode == 0 and len(scores) > 0 and docked_out.exists()

        return {
            "pep_id": pep_id,
            "success": success,
            "best_affinity": best_affinity,
            "all_scores": scores,
            "docked_pdbqt": docked_out if success else None,
            "log_file": log_out,
            "error": None if success else res.stderr
        }
    except Exception as e:
        return {
            "pep_id": pep_id,
            "success": False,
            "best_affinity": 0.0,
            "all_scores": [],
            "docked_pdbqt": None,
            "log_file": log_out,
            "error": str(e)
        }


def run_parallel_screening(
    ligands: Dict[str, Path],
    receptor_pdbqt: Path,
    grid: Optional[Dict[str, float]],
    out_dir: Path,
    threads: int = 4,
    max_workers: int = 2,
    exhaustiveness: int = 4,
    num_modes: int = 9,
    energy_range: float = 4.0,
    vina_bin: str = "vina",
    local_only: bool = False,
    autobox: bool = False,
    timeout: int = 900
) -> Dict[str, Dict[str, Any]]:
    """
    Distributes docking tasks across CPU workers using ProcessPoolExecutor.
    Caps concurrent workers to prevent memory thrashing on large protein grids.
    """
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    total_threads = threads if (threads and threads > 0) else (os.cpu_count() or 4)

    worker_cap = max_workers if (max_workers and max_workers > 0) else min(total_threads, 4)
    num_workers = max(1, min(total_threads, len(ligands), worker_cap))
    cpu_per_task = max(1, total_threads // num_workers)

    mode_label = "Local Induced-Fit Refinement" if local_only else f"Virtual Screening (exh={exhaustiveness})"
    logger.info(
        f"Starting {mode_label} of {len(ligands)} peptides on {num_workers} workers "
        f"({cpu_per_task} threads/worker, total CPU budget: {total_threads})..."
    )

    results = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_map = {
            executor.submit(
                dock_single_peptide,
                pep_id=pep_id,
                ligand_pdbqt=lig_path,
                receptor_pdbqt=receptor_pdbqt,
                grid=grid,
                out_dir=out_dir,
                vina_bin=vina_bin,
                exhaustiveness=exhaustiveness,
                num_modes=num_modes,
                energy_range=energy_range,
                cpu_per_task=cpu_per_task,
                local_only=local_only,
                autobox=autobox,
                timeout=timeout
            ): pep_id
            for pep_id, lig_path in ligands.items()
        }

        with tqdm(total=len(future_map), desc="Screening AMPs", unit="pep") as pbar:
            for future in as_completed(future_map):
                pep_id = future_map[future]
                try:
                    res = future.result()
                    results[pep_id] = res
                    status_str = f"{res['best_affinity']:.1f} kcal/mol" if res["success"] else "FAILED"
                    pbar.set_postfix({"last": f"{pep_id} ({status_str})"})
                except Exception as exc:
                    logger.error(f"Worker exception for {pep_id}: {exc}")
                    results[pep_id] = {"pep_id": pep_id, "success": False, "best_affinity": 0.0}
                finally:
                    pbar.update(1)

    successful_count = sum(1 for r in results.values() if r.get("success", False))
    logger.info(f"Screening complete: {successful_count}/{len(ligands)} docked successfully.")
    return results
