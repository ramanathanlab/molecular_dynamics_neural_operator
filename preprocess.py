import uuid
import jinja2
import subprocess
import numpy as np
from pathlib import Path
import MDAnalysis as mda
from gridData import Grid
from tqdm import tqdm
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    import numpy.typing as npt


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
    pdb_file: str, traj_file: str, scratch_dir: str
) -> "npt.ArrayLike":
    """Converts a trajectory file to an electrostatic grid."""
    scratch_dir = Path(scratch_dir)
    u = mda.Universe(str(pdb_file), str(traj_file))
    atoms = u.select_atoms("all")
    grids = []
    tmp_prefix = scratch_dir / str(uuid.uuid4())
    for _ in tqdm(u.trajectory):
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

    # Clean up temp files at the end
    tmp_pdb_file.unlink()
    tmp_pqr_file.unlink()
    tmp_log_file.unlink()
    tmp_in_file.unlink()
    tmp_dx_file.unlink()

    return np.array(grids)
