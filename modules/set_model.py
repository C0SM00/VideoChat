from modules.models.qwen import QwenModel


def setup_model(api_key: str):
    return QwenModel(api_key)
# todo: more parameters of model should be set here
