from typing import Any, Dict


def set_hyperparameter_ranges(
    cfg: Dict[str, Any], model_type: str, trial
) -> Dict[str, Any]:
    space = cfg["models"]["optimization_ranges"][model_type]

    hp: Dict[str, Any] = {}

    for param, spec in space.items():
        if spec["type"] == "int":
            hp[param] = trial.suggest_int(
                param,
                spec["min"],
                spec["max"],
                step=spec.get("step"),
            )

        elif spec["type"] == "float":
            hp[param] = trial.suggest_float(
                param,
                spec["min"],
                spec["max"],
                log=spec.get("log", False),
            )

        elif spec["type"] == "categorical":
            hp[param] = trial.suggest_categorical(
                param,
                spec["values"],
            )

        else:
            raise ValueError(f"Unknown hyperparameter type: {spec['type']}")

    return hp
