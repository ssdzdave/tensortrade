"""Allow `python -m daily_signals` to invoke the CLI."""

import sys

from daily_signals.cli import main

if __name__ == "__main__":
    sys.exit(main())
