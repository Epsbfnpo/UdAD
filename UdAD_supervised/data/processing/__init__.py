import importlib


def find_processing_using_name(processing_name):
    processing_filename = "data.processing." + processing_name + "_processing"
    processinglib = importlib.import_module(processing_filename)
    processing = None
    target_processing_name = processing_name.replace('_', '') + 'processing'

    # Iterate through all items in the module's dictionary to find a matching class.
    for name, cls in processinglib.__dict__.items():
        if name.lower() == target_processing_name.lower():
            processing = cls

    # If no matching class is found, print an error message and exit the program.
    if processing is None:
        print("In %s.py, there should be a dataprocessing with class name that matches %s in lowercase." % (
        processing_filename, target_processing_name))
        exit(0)

    return processing


def create_processing(processing_name, opt):
    processing_class = find_processing_using_name(processing_name)

    # Instantiate the processing class with the provided options.
    processing = processing_class(opt)

    return processing
