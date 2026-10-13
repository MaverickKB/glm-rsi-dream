"""Run the original TensorFold console entry point with the inference extension."""

import os
import sys
from importlib.metadata import entry_points

from .bootstrap import install
from .runtime import ENV


def main():
    original = list(entry_points(group="console_scripts", name="tensorfold"))
    if len(original) != 1:
        raise RuntimeError("Exactly one installed stock TensorFold entry point is required")
    if len(sys.argv) > 1 and sys.argv[1] == "serve" and "--help" not in sys.argv:
        if not os.environ.get(ENV):
            raise ValueError(ENV + " is required on both serving ranks")
        install()
    return original[0].load()()


if __name__ == "__main__":
    sys.exit(main())
