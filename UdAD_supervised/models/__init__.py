import importlib


def find_model_using_name(model_name):
    """
    Finds and returns the model class using the provided name.

    This function dynamically imports a module corresponding to the given model name and searches for a class within that module that matches the expected class name (formed by removing underscores from the model name and appending 'model').

    Args:
        model_name (str): The name of the model module to be imported and the class to be searched for.

    Returns:
        type: The model class that matches the given name.

    Raises:
        SystemExit: If no matching class is found, the program exits with an error message.
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'

    # Iterate over items in the module's dictionary to find a class matching the target model name.
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    # If no matching class is found, print an error message and exit the program.
    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """
    Retrieves the option setter method from the model class.

    This function finds the model class using the provided name and returns its `modify_commandline_options` method, which is used to set or modify command-line options specific to the model.

    Args:
        model_name (str): The name of the model class.

    Returns:
        function: The `modify_commandline_options` method from the model class.
    """
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """
    Creates and initializes an instance of the model based on the provided options.

    This function finds the appropriate model class using the model name specified in the options, creates an instance of the model, and prints a confirmation message indicating the model has been created.

    Args:
        opt (object): Configuration options containing the model name and other parameters.

    Returns:
        object: An instance of the model class, initialized with the provided options.
    """
    model_class = find_model_using_name(opt.model)
    instance = model_class(opt)
    return instance
