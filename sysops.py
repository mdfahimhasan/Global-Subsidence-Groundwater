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
    separator=[os.sep,'/']
    if directory_str[-1] not in separator:
        proper_dir_name=directory_str+os.sep
    return proper_dir_name

def makedirs(directory_list):
    """
    Make directory (if not exists) from a list o directory.
    
    Parameters:
    directory_list : A list of directories to create.
    Returns : 

    """
    for directory in directory_list:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory   
    


        