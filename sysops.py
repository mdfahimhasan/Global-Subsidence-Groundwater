import os


def make_proper_dir_name(directory_str):
    """
    Append os.sep to the last of directory name if not present.
    
    Parameters:
    directory_str : Directory path in string.
    
    Returns : Proper directory path with os.sep added to the end.
    """

    if directory_str is None:
        return None
    separator = [os.sep, '/']
    if directory_str[-1] not in separator:
        proper_dir_name = directory_str+os.sep
    return proper_dir_name


def makedirs(directory_list):
    """
    Make directory (if not exists) from a list o directory.
    
    Parameters:
    directory_list : A list of directories to create.

    Returns : A directory.
    """
    for directory in directory_list:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory   


def make_folderpath(maindir, path1, path2='', path3='', path4='', path5=''):
    """
    Join folder names to create a path directory string.

    Parameters:
    maindir : Relative or absolute path string of the main directory.
    path1 : Folder 1 to join.
    path2 : Folder 2 to join. Defaults to ''.
    path3 : Folder 3 to join. Defaults to ''.
    path4 : Folder 4 to join. Defaults to ''.
    path5 : Folder 5 to join. Defaults to ''.

    Returns : String containing folder path.
    """
    filepath = os.path.join(maindir, path1, path2, path3, path4, path5)
    return filepath
