from models.largeloss import LargeLossMatters

import utils.logging as logging

def get_model(config, num_classes):
    if config['LOGGING']['load_model']:
        model = logging.load_model(model, config)
        return model

    model_config = config['METHOD']

    if model_config['name'] == 'LargeLossMatters':
        try:
            model = LargeLossMatters(
                num_classes=num_classes,
                backbone=model_config['backbone'],
                freeze_backbone=model_config['freeze_backbone'],
                mod_schemes=model_config['mod_scheme'],
                delta_rel = model_config['delta_rel'],
            )
        except KeyError:
            raise KeyError("Please check your config file, requirements backbone, freeze_backbone, mod_schemes and delta_rel")
    else:
        raise NotImplementedError
    
    return model