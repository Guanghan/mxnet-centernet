from models.large_hourglass import get_large_hourglass_net

_model_factory = {
    'hourglass': get_large_hourglass_net,
}


def create_model(arch, heads, head_conv_channels):
    ind = arch.find('_')
    num_layers = int(arch[ind+1:]) if '_' in arch else 0
    arch = arch[:ind] if '_' in arch else arch

    get_model_func = _model_factory[arch]
    model = get_model_func(num_layers=num_layers, heads=heads, head_conv=head_conv_channels)
    return model

def load_model(model, model_load_path, ctx):
    model.load_parameters(model_load_path, ctx=ctx, ignore_extra=True)
    return model

def save_model(model, model_save_path):
    model.save_parameters(model_save_path)
