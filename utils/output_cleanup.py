from enum import Enum
from itertools import chain
from pathlib import Path
from typing import List, Set
from zipfile import ZIP_DEFLATED, ZipFile

from utils.output import OUT_DIR, remove_out_directory
from utils.progress_bar import ProgressBar


class Operation(Enum):
    CLEAN_UP = 0
    ARCHIVE = 1
    REMOVE = 2

    def __str__(self):
        return self.name


def clean_up(affected_dirs: List[Path]) -> None:
    """
    Remove log file, images and saved networks.

    :param affected_dirs: The list of directories to process.
    """

    nb_file_removed = 0
    for directory in affected_dirs:
        img_dir = directory / 'img'
        log_dir = directory / 'run.log'

        # Remove log
        if log_dir.exists():
            (directory / 'run.log').unlink()
            nb_file_removed += 1

        # Remove images
        if img_dir.is_dir():
            # Remove png and gif images files
            for image_file in chain(img_dir.glob('*.png'), img_dir.glob('*.gif'), img_dir.glob('*.mp4')):
                image_file.unlink()
                nb_file_removed += 1
            img_dir.rmdir()

        # Remove saved networks
        for p_file in directory.glob('*.pt'):
            nb_file_removed += 1
            p_file.unlink()

    print(f'Finish clean up {len(affected_dirs)} directories. {nb_file_removed} files successfully removed.')


def archive(affected_dirs: List[Path], out_path: Path) -> None:
    """
    Copy files into a zip archive. Remove nothing.
    Go to 1 directory deep only.

    :param affected_dirs: The list of directories to archive.
    :param out_path: The output zip file path.
    """

    if out_path.exists():
        raise FileExistsError(f'Output zip file already exist: {out_path}.')

    nb_file_archived = 0
    with ZipFile(out_path, 'w', ZIP_DEFLATED) as zf, ProgressBar(len(affected_dirs), task_name='Archiving') as progress:
        for directory in affected_dirs:
            for file_or_dir in directory.iterdir():
                # Copy file
                if file_or_dir.is_file():
                    zf.write(file_or_dir, arcname=file_or_dir.relative_to(Path(OUT_DIR)))
                    nb_file_archived += 1
                # Explore 1 level deep directory
                if file_or_dir.is_dir():
                    for file in file_or_dir.iterdir():
                        # Copy second level file
                        if file.is_file():
                            zf.write(file, arcname=file.relative_to(Path(OUT_DIR)))
                            nb_file_archived += 1
            progress.incr()

    print(f'Finish archive {nb_file_archived} files from {len(affected_dirs)} directories to "{out_path}"')


def remove(affected_dirs: List[Path]) -> None:
    """
    Remove files and directories.

    :param affected_dirs: The list of directories to remove recursively.
    """
    for directory in affected_dirs:
        remove_out_directory(directory)

    print(f'Finish remove {len(affected_dirs)} directories.')


def main(operations: Set[Operation], pattern: str) -> None:
    """
    Start the maintenance operations.

    :param operations: The list of operation to apply.
    :param pattern: The pattern to select files in the output directory
    """
    if len(operations) == 0:
        raise ValueError('Need at lest one operation.')

    runs_dir = Path(OUT_DIR)
    affected_dirs = list(runs_dir.glob(pattern))

    if len(affected_dirs) > 0:
        print(f'{len(affected_dirs)} output directories will be {", ".join(map(str, operations))} '
              f'from "{runs_dir.expanduser().resolve(strict=True)}/{pattern}".')
    else:
        print(f'No directory affected by "{runs_dir.expanduser().resolve(strict=True)}/{pattern}". Operation aborted.')
        exit(0)

    # Warning
    if Operation.REMOVE in operations:
        print(f"\033[1;31m[WARNING] This action can't be undone! "
              f"All {len(affected_dirs)} directories will be definitely removed!\033[0m")
    elif Operation.CLEAN_UP in operations:
        print("\033[0;33m[WARNING] This action can't be undone!\033[0m")

    validation = input('Continue this operation? [y/N] : ')

    if validation.lower() != 'y':
        print('Operation aborted.')
        exit(0)

    else:
        if Operation.CLEAN_UP in operations:
            clean_up(affected_dirs)

        if Operation.ARCHIVE in operations:
            file_name = 'out-archive' if pattern.strip() == '*' else pattern.replace('*', '').strip()
            archive(affected_dirs, runs_dir / f'{file_name}.zip')

        if Operation.REMOVE in operations:
            remove(affected_dirs)


if __name__ == '__main__':
    main(
        {
            # List of operations (CLEAN_UP, ARCHIVE and/or REMOVE)
            Operation.CLEAN_UP,
            Operation.ARCHIVE,
            # Operation.REMOVE
        },
        # Pattern to select files in the output directory
        'tmp'
    )
