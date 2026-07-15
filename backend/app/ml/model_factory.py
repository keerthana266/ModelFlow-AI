import importlib
import inspect
from typing import Any, Dict

from sklearn.base import BaseEstimator

from app.core.config import settings


class ModelFactoryError(ValueError):
    pass


def _as_dict(definition: Any) -> Dict[str, Any]:
    if isinstance(definition, dict):
        return definition

    return {
        "model_zoo_id": getattr(definition, "model_zoo_id", None),
        "name": getattr(definition, "name", None),
        "framework": getattr(definition, "framework", None),
        "task_type": getattr(definition, "task_type", None),
        "class_path": getattr(definition, "class_path", None),
        "default_parameters": getattr(definition, "default_parameters", {}) or {},
        "hyperparameter_space": getattr(
            definition,
            "hyperparameter_space",
            {},
        )
        or {},
    }


def _validate_class_path(class_path: str) -> tuple[str, str]:
    if not class_path or "." not in class_path:
        raise ModelFactoryError(
            "class_path must contain a full module and class name."
        )

    module_name, class_name = class_path.rsplit(".", 1)
    root_module = module_name.split(".", 1)[0]

    if root_module not in settings.model_zoo_allowed_modules_list:
        allowed = ", ".join(settings.model_zoo_allowed_modules_list)
        raise ModelFactoryError(
            f"Module '{root_module}' is not allowed. Allowed modules: {allowed}."
        )

    path_parts = module_name.split(".") + [class_name]

    if any(not part or part.startswith("_") for part in path_parts):
        raise ModelFactoryError(
            "Private modules, private classes, and empty path segments are not allowed."
        )

    return module_name, class_name


def load_model_class(class_path: str):
    module_name, class_name = _validate_class_path(class_path)

    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise ModelFactoryError(
            f"Could not import module '{module_name}': {error}"
        ) from error

    try:
        model_class = getattr(module, class_name)
    except AttributeError as error:
        raise ModelFactoryError(
            f"Class '{class_name}' was not found in module '{module_name}'."
        ) from error

    if not inspect.isclass(model_class):
        raise ModelFactoryError(f"'{class_path}' is not a Python class.")

    if not issubclass(model_class, BaseEstimator):
        raise ModelFactoryError(
            "Only scikit-learn compatible estimators derived from BaseEstimator "
            "are supported."
        )

    return model_class


def create_model_instance(
    definition: Any,
    parameter_overrides: Dict[str, Any] | None = None,
):
    payload = _as_dict(definition)
    class_path = str(payload.get("class_path") or "")
    model_class = load_model_class(class_path)

    parameters = dict(payload.get("default_parameters") or {})
    parameters.update(parameter_overrides or {})

    try:
        model = model_class(**parameters)
    except Exception as error:
        raise ModelFactoryError(
            f"Could not instantiate '{class_path}' with the configured parameters: "
            f"{error}"
        ) from error

    if not callable(getattr(model, "fit", None)):
        raise ModelFactoryError("The configured estimator does not implement fit().")

    if not callable(getattr(model, "predict", None)):
        raise ModelFactoryError(
            "The configured estimator does not implement predict()."
        )

    return model


def validate_model_definition(definition: Any) -> Dict[str, Any]:
    payload = _as_dict(definition)
    model = create_model_instance(payload)

    try:
        resolved_parameters = model.get_params(deep=False)
    except Exception as error:
        raise ModelFactoryError(
            f"The estimator does not expose valid scikit-learn parameters: {error}"
        ) from error

    search_space = payload.get("hyperparameter_space") or {}

    for parameter_name in search_space:
        if parameter_name not in resolved_parameters:
            raise ModelFactoryError(
                f"Hyperparameter '{parameter_name}' is not accepted by "
                f"'{payload.get('class_path')}'."
            )

    return {
        "valid": True,
        "class_path": payload.get("class_path"),
        "resolved_class": (
            f"{model.__class__.__module__}.{model.__class__.__name__}"
        ),
        "default_parameter_count": len(payload.get("default_parameters") or {}),
        "search_parameter_count": len(search_space),
        "supports_predict_proba": callable(
            getattr(model, "predict_proba", None)
        ),
    }


def suggest_hyperparameters(
    trial,
    hyperparameter_space: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {}

    for parameter_name, raw_config in (hyperparameter_space or {}).items():
        config = dict(raw_config)
        parameter_type = config.get("type")

        if parameter_type == "integer":
            low = int(config["low"])
            high = int(config["high"])
            log = bool(config.get("log", False))

            if log:
                parameters[parameter_name] = trial.suggest_int(
                    parameter_name,
                    low,
                    high,
                    log=True,
                )
            else:
                parameters[parameter_name] = trial.suggest_int(
                    parameter_name,
                    low,
                    high,
                    step=int(config.get("step") or 1),
                )

        elif parameter_type == "float":
            low = float(config["low"])
            high = float(config["high"])
            log = bool(config.get("log", False))
            step = config.get("step")

            if log:
                parameters[parameter_name] = trial.suggest_float(
                    parameter_name,
                    low,
                    high,
                    log=True,
                )
            elif step is not None:
                parameters[parameter_name] = trial.suggest_float(
                    parameter_name,
                    low,
                    high,
                    step=float(step),
                )
            else:
                parameters[parameter_name] = trial.suggest_float(
                    parameter_name,
                    low,
                    high,
                )

        elif parameter_type == "categorical":
            parameters[parameter_name] = trial.suggest_categorical(
                parameter_name,
                list(config.get("choices") or []),
            )

        else:
            raise ModelFactoryError(
                f"Unsupported hyperparameter type '{parameter_type}' "
                f"for '{parameter_name}'."
            )

    return parameters
