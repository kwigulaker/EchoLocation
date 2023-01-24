"""
Function for monitoring a source directory, and copying all new files to a destination directory.
"""
import os
import sys
import click
import shutil
import time
from tqdm import tqdm
from pathlib import Path
from functools import partial
# Own imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_functions"))
from file_handling import _file_log_init, check_exclusive_lock, find_new_or_changed_files, create_dest_path

def monitor(src, dst, timeout, ext, logging_level, n_subfolders=1, filter:list=None):
    
    # Fix paths
    src = Path(os.path.realpath(src))
    assert src.is_dir(), f"Data folder '{src}' do not exist!"
    dst = Path(os.path.realpath(dst))
    os.makedirs(dst, exist_ok=True)
    FILE_LOG = _file_log_init(dst, ext)
        
    # Run until stopped by ctrl + c, listening on folder and run when files appear
    try:
        while True:
            new_or_changed = find_new_or_changed_files(src, file_log=FILE_LOG, ext=f"{ext if ext else ''}")
            if new_or_changed:
                files = [f for f in new_or_changed if os.path.isfile(f) and "9999.kmall" not in f and not check_exclusive_lock(f)]
                if filter:
                    files = [f for f in files if any(map(f.__contains__, [str(Path(f).parents[n_subfolders] / x) for x in filter]))]
                print(f"Found {len(files)} new files, moving to processing queue ...")
                for file in tqdm(files):
                    dest_path = create_dest_path(file, dst, n_subfolders=n_subfolders)
                    shutil.copy(file, dest_path)
            else:
                if timeout == -1:
                    break
                if logging_level > 1:
                    print(f"Awaiting files, sleeping for {timeout} seconds")
                time.sleep(timeout)       
    except KeyboardInterrupt:
        pass

click.option = partial(click.option, show_default=True)
@click.command()
@click.argument("src", nargs=1)
@click.argument("dst", nargs=1)
@click.option('-t', '--timeout', default=60, type=int, help='Timeout between checks.')
@click.option('-e', '--ext', default=".kmall", help='Extension of files.')
@click.option("-l", '--logging_level', default=0, help="Level of verbosity. 0 includes bare minimum, 1 includes additional information and 2 includes data just for convenience.")
def main(src, dst, timeout, ext, logging_level):
    monitor(src, dst, timeout, ext, logging_level, n_subfolders=2)

if __name__ == '__main__':
    main()