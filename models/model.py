from .networks.large_hourglass import get_large_hourglass_net

_model_factory = {
    'hourglass': get_large_hourglass_net,
}


def create_mode():
    return

def load_model():
    return

def save_model():
    return