import uuid
import jinja2
import subprocess
import numpy as np
from pathlib import Path
import MDAnalysis as mda
from gridData import Grid
from tqdm import tqdm
from typing import TYPE_CHECKING, TextIO, Union, List
from concurrent.futures import ProcessPoolExecutor

if TYPE_CHECKING:
    import numpy.typing as npt

PathLike = Union[Path, str]


def run_subprocess(executable: str, stdout: TextIO, cwd: Path) -> int:
    completed_proc = subprocess.run(
        f"{executable}",
        shell=True,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        encoding="utf-8",
    )
    return completed_proc.returncode


def write_in_file(in_file: Path, pqr_file: Path, dx_file: Path) -> None:
    file_loader = jinja2.FileSystemLoader("templates")
    env = jinja2.Environment(loader=file_loader)
    template = env.get_template("electrostatics.j2")
    contents = template.render(pqr_file=pqr_file, dx_file=dx_file.with_suffix(""))
    with open(in_file, "w") as f:
        f.write(contents)


def trajectory_to_electrostatic_grid(
    pdb_file: PathLike,
    traj_file: PathLike,
    scratch_dir: PathLike,
    results_dir: PathLike,
    verbose: bool = False,
    save_interval: int = 1000,
) -> None:
    """Converts a trajectory file to an electrostatic grid."""
    scratch_dir = Path(scratch_dir)
    results_dir = Path(results_dir)
    u = mda.Universe(str(pdb_file), str(traj_file))
    atoms = u.select_atoms("all")
    grids = []
    tmp_prefix = scratch_dir / str(uuid.uuid4())
    iterable = tqdm(enumerate(u.trajectory)) if verbose else enumerate(u.trajectory)

    for i, _ in iterable:
        tmp_pdb_file = tmp_prefix.with_suffix(".pdb")
        tmp_pqr_file = tmp_prefix.with_suffix(".pqr")
        tmp_log_file = tmp_prefix.with_suffix(".log")
        tmp_in_file = tmp_prefix.with_suffix(".in")
        tmp_dx_file = tmp_prefix.with_suffix(".dx")

        atoms.write(tmp_pdb_file)
        with open(tmp_log_file, "w") as stdout:
            pbd2pqr_exec = f"pdb2pqr30 {tmp_pdb_file} {tmp_pqr_file}"
            retcode = run_subprocess(pbd2pqr_exec, stdout, scratch_dir)
            if retcode != 0:
                raise ValueError(f"pbd2pqr_exec failed with return code: {retcode}")

        write_in_file(tmp_in_file, tmp_pqr_file, tmp_dx_file)

        with open(tmp_log_file, "w") as stdout:
            apbs_exec = f"apbs {tmp_in_file}"
            retcode = run_subprocess(apbs_exec, stdout, scratch_dir)
            if retcode != 0:
                raise ValueError(f"apbs failed with return code: {retcode}")

        # Parse dx file into np.ndarray containing the grid
        grids.append(Grid(str(tmp_dx_file)).grid)

        if i % save_interval == 0:
            npy_file = results_dir / (Path(traj_file).with_suffix("").name + f"_{i}.npy")
            print(npy_file)
            np.save(npy_file, np.array(grids))
            del grids
            grids = []

    # Clean up any remainder
    if grids:
        npy_file = results_dir / (Path(traj_file).with_suffix("").name + f"_{i}.npy")
        np.save(npy_file, np.array(grids))
        del grids
        grids = []

    # Clean up temp files at the end
    tmp_pdb_file.unlink()
    tmp_pqr_file.unlink()
    tmp_log_file.unlink()
    tmp_in_file.unlink()
    tmp_dx_file.unlink()


def _worker(kwargs) -> None:
    """Helper function for parallel data preprocessing."""
    return trajectory_to_electrostatic_grid(**kwargs)


def parallel_trajectory_to_electrostatic_grid(
    pdb_files: List[PathLike],
    traj_files: List[PathLike],
    scratch_dir: PathLike,
    results_dir: PathLike,
    num_workers: int = 10,
    save_interval: int = 1000,
) -> None:

    kwargs = [
        {
            "pdb_file": pdb_file,
            "traj_file": traj_file,
            "scratch_dir": scratch_dir,
            "results_dir": results_dir,
            "save_interval": save_interval,
            "verbose": bool(i % num_workers == 0),
        }
        for i, (pdb_file, traj_file) in enumerate(zip(pdb_files, traj_files))
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in executor.map(_worker, kwargs):
            pass
