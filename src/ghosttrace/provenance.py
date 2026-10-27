"""Source provenance includes uncommitted work; a Git HEAD alone is insufficient."""
import subprocess
from pathlib import Path
from ghosttrace.config import digest,file_digest

def code_provenance():
    package=Path(__file__).resolve().parent
    files={str(path.relative_to(package)):file_digest(path) for path in sorted(package.rglob('*.py'))}
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True,stderr=subprocess.DEVNULL).strip())
    except (OSError,subprocess.CalledProcessError):commit=None;dirty=None
    return {'git_commit':commit,'workspace_dirty':dirty,'source_sha256':digest(files),'source_file_sha256':files}
