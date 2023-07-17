import shutil
from pathlib import Path


def unzip_directory(directory_path):
    directory = Path(directory_path)
    for file in directory.glob("*.zip"):
        shutil.unpack_archive(str(file), extract_dir=str(file.parent))
