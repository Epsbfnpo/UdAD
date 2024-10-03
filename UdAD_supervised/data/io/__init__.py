import importlib


def find_io_using_name(io_name):
    io_filename = "data.io." + io_name + "_io"
    iolib = importlib.import_module(io_filename)
    io = None
    target_io_name = io_name.replace('_', '') + 'io'

    for name, cls in iolib.__dict__.items():
        if name.lower() == target_io_name.lower():
            io = cls

    if io is None:
        print("In %s.py, there should be a dataio with class name that matches %s in lowercase." % (io_filename, target_io_name))
        exit(0)

    return io


def create_io(io_name, opt):
    io_class = find_io_using_name(io_name)

    io = io_class(opt)

    return io