import os
import pickle


def mkdirs(paths):
    """
    Creates directories for the specified paths.

    This function can handle both a single path or a list of paths. It ensures that the directory exists, creating it if necessary.

    Args:
        paths (str or list of str): A single path or a list of paths for which directories should be created.
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """
    Creates a directory if it does not already exist.

    Args:
        path (str): The directory path to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def list_dir(directory):
    """
    Lists all non-hidden files in the specified directory.

    This function returns a list of all files in the directory that do not start with a '.' (hidden files).

    Args:
        directory (str): The directory path to list files from.

    Returns:
        list of str: A list of non-hidden file names in the directory.
    """
    visible_files = []
    for file in os.listdir(directory):
        if not file.startswith('.'):
            visible_files.append(file)
    return visible_files


def save_pickle(file, path):
    """
    Saves an object to a file using the pickle protocol.

    Args:
        file (object): The object to be serialized and saved.
        path (str): The file path where the object should be saved.
    """
    with open(path, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    """
    Reads and deserializes a pickle file from the specified path.

    Args:
        path (str): The file path from which the object should be loaded.

    Returns:
        object: The object deserialized from the pickle file.
    """
    with open(path, 'rb') as handle:
        pickle_file = pickle.load(handle)
    return pickle_file
