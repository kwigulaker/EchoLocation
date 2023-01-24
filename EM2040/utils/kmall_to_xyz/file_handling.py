"""
Utility functions for file handling, being:
- Finding distinct filenames
- Finding new or changed files
- Copying files
- Checking exclusive locks
- Creating destination paths based on relative paths.
"""
import os
import sys
import shutil
from glob import glob
from pathlib import Path

###############################
##  Find distinct filenames  ##
###############################
def find_distinct_filenames(dir1:str, dir2:str, ext:str=None) -> tuple:
    """
    Function for finding distinct or common files on one location over the other, 
    simply based on the name of the file (relative path match)
    """
    left_files = set(os.path.relpath(x, dir1) for x in glob(os.path.join(dir1, f"**/*{ext if ext else ''}"), recursive=True))
    right_files = set(os.path.relpath(x, dir2) for x in glob(os.path.join(dir2, f"**/*{ext if ext else ''}"), recursive=True))
    left_only = [os.path.join(dir1, file) for file in left_files.difference(right_files)]
    right_only = [os.path.join(dir2, file) for file in right_files.difference(left_files)]
    common = list(left_files.intersection(right_files))
    return left_only, right_only, common

##################################
##  Find new or modified files  ##
##################################
def _file_log_init(data_dir, ext=None) -> dict:
    """
    Create a log of new or modified files, either all or with given extension, recursively, based on modified time
    """
    FILE_LOG = dict()
    for file in glob(os.path.join(data_dir, f"**/*{ext if ext else ''}"), recursive=True):
        FILE_LOG[file] = None
    return FILE_LOG

def find_new_or_changed_files(data_dir:str, file_log:dict, ext:str=None) -> list:
    """
    Find files which is new or changed since last time you checked, according to the log FILE_LOG
    """
    files = []
    for file in glob(os.path.join(data_dir, f"**/*{ext if ext else ''}"), recursive=True):
        if file not in file_log:
            files.append(file)
            file_log[file] = os.path.getmtime(file)
        else:    
            if file_log[file] != os.path.getmtime(file):
                files.append(file)
                file_log[file] = os.path.getmtime(file)
    return files

##########################
##  Create dest folder  ##
##########################
def create_dest_path(file:str, out_dir:str, n_subfolders:int=1) -> str:
    """Get new path from file based on number of subfolders (data/4095/4095.tiff has one subfolder from data to file)

    Args:
        file (str): file path
        out_dir (str): output directory path
        n_subfolders (int, optional): number of subfolders from out_path. Defaults to 1.
    Returns:
        str: destination path
    """
    relpath = os.path.relpath(file, Path(file).parents[n_subfolders].absolute())
    dest_path = Path(out_dir) / relpath
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    return dest_path


########################
##  Simple file copy  ##
########################
def copy_file(file:str, out_dir:str, n_subfolders:int=1):
    """
    Simply function for copying file from data dir to out dir
    """
    dest_path = create_dest_path(file, out_dir, n_subfolders=n_subfolders)
    shutil.copy(file, dest_path)
    return str(dest_path)


########################
##   File locking v1  ##
########################
def check_exclusive_lock(fpath:str):
    """Check if file is locked exclusively or not

    Args:
        fpath (str): path to file

    Returns:
        Bool: answer to "is the file locked?"
    """
    try: # open
        if not os.path.isfile:
            raise FileNotFoundError
        os.rename(fpath, fpath)
        return False
    except FileNotFoundError:
        print("File do not exist!")
        sys.exit(0)
    except OSError as e: # Locked
        return True
