from zenml import step
import importlib

@step
def dynamic_model_loader_step(model_path: str, import_path: str):
    """
    Dynamically import a model loader and load the model from the given path.
    Example import_path: 'joblib:load'
    """
    if ':' in import_path:
        module_path, attr_name = import_path.split(':')
    elif '.' in import_path:
        module_path, attr_name = import_path.rsplit('.', 1)
    else:
        raise ValueError(f"Invalid import path: {import_path}")
    module = importlib.import_module(module_path)
    loader = getattr(module, attr_name)
    return loader(model_path) 