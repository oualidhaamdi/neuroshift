"""
FSL Wrapper - Executes FSL commands via WSL2.
Handles Windows-WSL path conversion and space-in-path issues.
"""

import subprocess
import sys
import os
import hashlib
import shlex
import time
from pathlib import Path
import SimpleITK as sitk
from src.core import setup_logger

logger = setup_logger(__name__)

def win_to_wsl(win_path: Path | str) -> str:
    """Converts Windows path to a WSL-safe path, avoiding spaces via /tmp symlinks."""
    # NB: os.path.abspath (not .resolve()) so we do NOT follow NTFS junctions.
    # Data dirs (RunBatch_Output, outputs, ...) may be junctions to another
    # location whose real path contains spaces. Following them would bypass the
    # /tmp/NeuroShift_Workspace symlink and re-introduce spaces that FSL rejects.
    # Keeping the in-project path lets WSL resolve the junction itself.
    p = Path(os.path.abspath(win_path))

    # Check if path is within project to use project-level symlink
    curr = p
    proj_root = None
    for _ in range(10):
        if (curr / "src").exists() and (curr / "config.py").exists():
            proj_root = curr
            break
        if curr.parent == curr: break
        curr = curr.parent

    if proj_root and p.is_relative_to(proj_root):
        rel = p.relative_to(proj_root).as_posix()
        return f"/tmp/NeuroShift_Workspace/{rel}"
        
    # Fallback if outside project
    parts = list(p.parts)
    drive = parts[0].rstrip(':\\').lower()
    rest  = '/'.join(parts[1:]).replace('\\', '/')
    return f"/mnt/{drive}/{rest}"

class FSLWrapper:
    """Bridge for FSL commands through WSL2."""

    def __init__(self, wsl_distro: str | None = "Ubuntu", 
                 fsl_dir: str = "$HOME/miniforge3", timeout: int = 3600):
        self.timeout = timeout
        self.fsl_dir = fsl_dir
        self._distro = wsl_distro
        self._fsl_ok = None

    def _check(self):
        if self._fsl_ok: return
        
        # 1. Setup project workspace symlink in WSL
        target_path = Path(__file__).parents[1].resolve()
        parts = list(target_path.parts)
        drive = parts[0].rstrip(':\\').lower()
        rest  = '/'.join(parts[1:]).replace('\\', '/')
        real_wsl_path = f"/mnt/{drive}/{rest}"
        
        subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-c", 
                       f"mkdir -p /tmp && ln -sfn {shlex.quote(real_wsl_path)} /tmp/NeuroShift_Workspace"], 
                       capture_output=True)
        
        # 2. Advanced FSL Discovery
        # We try to find where 'bet' is actually located
        find_cmd = "which bet || ls /usr/local/fsl/bin/bet /opt/fsl/bin/bet $HOME/miniforge3/bin/bet $HOME/fsl/bin/bet 2>/dev/null | head -n 1"
        r_find = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-c", find_cmd], 
                                capture_output=True, text=True)
        
        found_path = r_find.stdout.strip()
        if not found_path:
            raise RuntimeError("FSL executable 'bet' not found in WSL. Please ensure FSL or miniforge3 is installed and in PATH.")
        
        # Infer FSLDIR from found path (e.g., /home/user/miniforge3/bin/bet -> /home/user/miniforge3)
        self.fsl_dir = str(Path(found_path).parent.parent).replace("\\", "/")
        logger.info(f"FSL discovered at: {self.fsl_dir}")
        
        self._fsl_ok = True

    def _wait_for_file_wsl(self, wsl_path: str, timeout=30):
        """Wait until file is visible AND has a stable (non-zero) size in WSL2.

        Checking existence alone is not enough: WSL2's 9P driver can report a
        file as present before Windows has finished flushing the write buffer,
        causing FSL to read a truncated/invalid NIfTI. We poll until the size
        has been non-zero and unchanged for two consecutive 0.5 s ticks.
        """
        start = time.time()
        prev_size = -1
        stable_count = 0
        while time.time() - start < timeout:
            check_cmd = ["wsl"]
            if self._distro:
                check_cmd += ["-d", self._distro]
            stat_r = subprocess.run(
                check_cmd + ["bash", "-c",
                             f"stat -c %s {shlex.quote(wsl_path)} 2>/dev/null || echo 0"],
                capture_output=True, text=True, check=False,
            )
            try:
                size = int(stat_r.stdout.strip())
            except ValueError:
                size = 0
            if size > 0 and size == prev_size:
                stable_count += 1
                if stable_count >= 2:   # two consecutive equal readings → stable
                    return True
            else:
                stable_count = 0
            prev_size = size
            time.sleep(0.5)
        return False

    def _run_wsl(self, cmd: list, check: bool = True):
        """Execute command in the correct WSL distribution with proper environment."""
        wsl_cmd = ["wsl"]
        if self._distro:
            wsl_cmd += ["-d", self._distro]
        
        # Poll for input file visibility if it's a known Workspace path
        for arg in cmd:
            arg_str = str(arg)
            if "/tmp/NeuroShift_Workspace" in arg_str and not arg_str.startswith("-"):
                if "." in arg_str: # Likely a file
                    self._wait_for_file_wsl(arg_str)

        cmd_quoted = [shlex.quote(str(c)) for c in cmd]
        env_vars = (
            f"export FSLDIR={self.fsl_dir} && "
            f"export FSLOUTPUTTYPE=NIFTI_GZ && "
            f"export PATH=\"{self.fsl_dir}/bin:/usr/local/bin:/usr/bin:/bin:$PATH\""
        )
        shell_cmd = f"{env_vars} && {' '.join(cmd_quoted)} && sync"
        wsl_cmd += ["bash", "-c", shell_cmd]
        
        r = subprocess.run(wsl_cmd, capture_output=True, text=True, check=False, encoding="utf-8")
        
        if check and r.returncode != 0:
            err_msg = r.stderr.strip() or r.stdout.strip() or "Unknown FSL error"
            raise RuntimeError(f"FSL Error (code {r.returncode}): {err_msg}")
        return r

    def bet(self, input_path: Path, output_path: Path, frac: float = 0.5, mask: bool = True):
        self._check()
        in_wsl = win_to_wsl(input_path)
        out_wsl = win_to_wsl(output_path)
        
        flags = ["-f", f"{frac:.2f}"]
        if mask: flags.append("-m")
        
        logger.info(f"  [FSL] BET: {input_path.name}")
        self._run_wsl(["bet", in_wsl, out_wsl] + flags)
        return Path(str(output_path) + ".nii.gz")

    def fast(self, input_path: Path, output_prefix: Path, n_classes: int = 3):
        self._check()
        in_wsl = win_to_wsl(input_path)
        out_wsl = win_to_wsl(output_prefix)
        
        logger.info(f"  [FSL] FAST: {input_path.name}")
        self._run_wsl(["fast", "-n", str(n_classes), "-t", "1", "--nopve", "-o", out_wsl, in_wsl])
        return Path(str(output_prefix) + "_seg.nii.gz")

    def flirt(self, src: Path, ref: Path, output: Path, dof: int = 12, 
              cost: str = "mutualinfo", omat: Path | None = None,
              inweight: Path | None = None, refweight: Path | None = None):
        self._check()
        src_wsl = win_to_wsl(src)
        ref_wsl = win_to_wsl(ref)
        out_wsl = win_to_wsl(output)
        
        cmd = ["flirt", "-in", src_wsl, "-ref", ref_wsl, "-out", out_wsl, 
               "-dof", str(dof), "-cost", cost, "-nosearch"]
        
        if omat: cmd += ["-omat", win_to_wsl(omat)]
        if inweight: cmd += ["-inweight", win_to_wsl(inweight)]
        if refweight: cmd += ["-refweight", win_to_wsl(refweight)]
        
        logger.info(f"  [FSL] FLIRT: {src.name} (DOF={dof})")
        self._run_wsl(cmd)
        return output

def check_fsl_availability() -> tuple[bool, str]:
    """Checks if FSL is installed and reachable via WSL2."""
    import subprocess
    try:
        # Check WSL
        subprocess.run(["wsl", "--list", "--quiet"], capture_output=True, timeout=5)
        
        # Check BET in standard locations
        cmd = ["wsl", "-d", "Ubuntu", "--", "bash", "-c", 
               "if [ -f $HOME/miniforge3/bin/bet ] || [ -f /usr/local/fsl/bin/bet ]; then echo FOUND; fi"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if "FOUND" in r.stdout:
            return True, "FSL found in WSL"
        return False, "FSL not found in WSL paths"
    except Exception as e:
        return False, f"WSL/FSL check failed: {e}"
