



def remove_layers(state_dict, layers):
    removed_layers = []
    temp_state_dict = state_dict.copy()

    if isinstance(layers, str):
        _layer = []
        _layer.append(layers)
        layers = _layer

    for key, value in temp_state_dict.items():
        for prefix in layers:
            if key.startswith(prefix):
                removed_layers.append(key)
                del state_dict[key]

    return state_dict, removed_layers


