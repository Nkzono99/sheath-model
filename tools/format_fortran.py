"""Apply whitespace and F2008 submodule-aware indentation in one atomic pass.

Use through pre-commit so both formatter versions come from the pinned hook
configuration. Build products and archived verification snapshots are excluded
by that configuration.
"""

from pathlib import Path
import os
import subprocess
import sys


def main():
    environment = os.environ.copy()
    environment.pop("FINDENT_FLAGS", None)
    for name in sys.argv[1:]:
        path = Path(name)
        original = path.read_bytes()
        whitespace = subprocess.run(
            ["fprettify", "--disable-indent", "--enable-decl", "--line-length", "132", "--stdout", str(path)],
            check=True, stdout=subprocess.PIPE,
        ).stdout
        formatted = subprocess.run(
            ["findent", "-ifree", "-ofree", "-i2", "-c2", "-C2", "-k4", "--align_paren=0"],
            input=whitespace, check=True, stdout=subprocess.PIPE, env=environment,
        ).stdout
        if original != formatted:
            path.write_bytes(formatted)
            print(f"Formatted {path}")


if __name__ == "__main__":
    main()
