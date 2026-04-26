import importlib

def dynamic_import(import_path: str):
    """
    Dynamically import a function or class from a string import path.
    Example: 'module.submodule:function_name'
    """
    if ':' in import_path:
        module_path, attr_name = import_path.split(':')
    elif '.' in import_path:
        # fallback: last dot is attribute
        module_path, attr_name = import_path.rsplit('.', 1)
    else:
        raise ValueError(f"Invalid import path: {import_path}")
    module = importlib.import_module(module_path)
    return getattr(module, attr_name) 