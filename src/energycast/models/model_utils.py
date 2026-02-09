from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_CONFIG_DIR = ROOT_DIR / "configs" / "models"


@dataclass
class ModelConfig:
    # data
    train_set: str
    dev_set: str
    test_set: str
    id_col: str
    time_col: str

    # model
    model_type: str
    strategy: str
    horizon: int
    target_col: str
    features: list[str] | None = None
    model_params: dict | None = None

    # tracking
    mlflow_experiment: str | None = None
    mlflow_run_name: str | None = None
    artifact_category: str | None = None
    artifact_name: str | None = None


def load_model_config(config_name: str) -> ModelConfig:
    """Loads model configuration from configs/models/

    Parameters
    ----------
    config_name : str
       name of the yaml configuration file
       should be provided without .yaml

    Returns
    -------
    ModelConfig
        dataclass object with model configuration
    """
    cfg = yaml.safe_load(
        Path(MODEL_CONFIG_DIR / f"{config_name}.yaml").read_text(
            encoding="utf-8"
        )
    )

    return ModelConfig(
        train_set=(cfg["data"]["train_set"]),
        dev_set=(cfg["data"]["dev_set"]),
        test_set=(cfg["data"]["test_set"]),
        id_col=cfg["data"]["id_col"],
        time_col=cfg["data"]["time_col"],
        model_type=cfg["model"]["type"],
        strategy=cfg["model"]["strategy"],
        horizon=cfg["model"]["horizon"],
        target_col=cfg["target"]["col"],
        features=cfg.get("features", {}).get("list") or [],
        model_params=cfg.get("lgbm_params") or {},
        mlflow_experiment=cfg["tracking"]["mlflow_experiment"],
        mlflow_run_name=cfg["tracking"]["mlflow_run_name"],
        artifact_category=cfg["tracking"]["artifact_category"],
        artifact_name=cfg["tracking"]["artifact_name"],
    )
