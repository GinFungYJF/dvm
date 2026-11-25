"""Check that the public archive symbols survive shared linkage (including templates)."""
import argparse
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("directory", nargs="?", default=".")
args = parser.parse_args()
root = Path(args.directory)


def symbols(path, dynamic=False):
    command = ["nm", "-g", "--defined-only", "--format=posix"]
    if dynamic:
        command.append("-D")
    output = subprocess.check_output(command + [str(path)], text=True)
    return {line.split()[0].split("@")[0] for line in output.splitlines() if line.split()}


archive = symbols(root / "libdvm.a")
shared = symbols(root / "libdvm.so.1", True)
# Both const and non-const member functions; no hard-coded instance count.
public = re.compile(r"^_ZNK?3dvm(?:6Kernel|6Config|4Comm|7Float16|8BFloat16)")
expected = {s for s in archive if public.match(s)}
missing = expected - shared
if missing:
    raise SystemExit("Missing shared API symbols:\n" + "\n".join(sorted(missing)))
if len(expected) < 200:
    raise SystemExit("Unexpectedly small public symbol set")
print(f"Verified {len(expected)} public symbols, including all instantiated templates")
# The Python binding resolves its DVM references from the static archive.
binding = root / "_dvm_py.so"
if binding.exists():
    output = subprocess.check_output(["nm", "-D", "--undefined-only", "--format=posix", str(binding)], text=True)
    refs = {line.split()[0].split("@")[0] for line in output.splitlines() if line.split()}
    missing = {s for s in refs if re.match(r"^_Z.*3dvm", s)}
    if missing:
        raise SystemExit("Unresolved binding symbols:\n" + "\n".join(sorted(missing)))
    print("Verified Python binding DVM references")
