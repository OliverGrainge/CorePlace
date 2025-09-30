from omegaconf import OmegaConf


def load_config(config_path: str):
    config = OmegaConf.load(config_path)
    return config
