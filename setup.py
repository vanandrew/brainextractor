import sys
import site
from pathlib import Path
from setuptools import setup

# This line enables user based installation when using pip in editable mode with the latest
# pyproject.toml config
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]
THISDIR = Path(__file__).parent

# get scripts path
scripts_path = THISDIR / "brainextractor" / "scripts"

setup(
    entry_points={
        "console_scripts": [
            f"{f.stem}=brainextractor.scripts.{f.stem}:main"
            for f in scripts_path.glob("*.py")
            if f.name not in "__init__.py"
        ]
    }
)
