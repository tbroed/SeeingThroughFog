import os
import yaml
import socket

from pathlib import Path
from easydict import EasyDict
from packaging import version


def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, 'NotFoundKey: %s' % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'NotFoundKey: %s' % subkey
        try:
            value = literal_eval(v)
        except:
            value = v

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            d[subkey] = value


def merge_new_config(config, new_config):

    if '_BASE_CONFIG_' in new_config:

        try:

            with open(new_config['_BASE_CONFIG_'], 'r') as f:

                if version.parse(yaml.__version__) < version.parse('5.1'):
                    yaml_config = yaml.load(f)
                else:
                    yaml_config = yaml.load(f, Loader=yaml.FullLoader)

        except FileNotFoundError:

            try:

                with open(str(new_config['_BASE_CONFIG_']).replace("../", ""), 'r') as f:

                    if version.parse(yaml.__version__) < version.parse('5.1'):
                        yaml_config = yaml.load(f)
                    else:
                        yaml_config = yaml.load(f, Loader=yaml.FullLoader)

            except FileNotFoundError:

                with open('../tools/' + str(new_config['_BASE_CONFIG_']), 'r') as f:

                    if version.parse(yaml.__version__) < version.parse('5.1'):
                        yaml_config = yaml.load(f)
                    else:
                        yaml_config = yaml.load(f, Loader=yaml.FullLoader)

        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file, config):

    with open(cfg_file, 'r') as f:

        if version.parse(yaml.__version__) < version.parse('5.1'):
            new_config = yaml.load(f)
        else:
            new_config = yaml.load(f, Loader=yaml.FullLoader)

        merge_new_config(config=config, new_config=new_config)

    return config


cfg = EasyDict()
# cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0

hostname = socket.gethostname()

if ('biwirender' in hostname) or ('bmicgpu' in hostname):           # training on CVL cluster

    cfg.ROOT_DIR = Path(f'/scratch_net/hox/mhahner/repositories/PCDet')

elif 'beast' in hostname:                                           # home office

    cfg.ROOT_DIR = Path(f'/home/mhahner/repositories/PCDet')

elif 'eu-' in hostname:                                             # euler

    cfg.ROOT_DIR = Path(os.environ['CODE_DIR'])

else:                                                               # AWS

    cfg.ROOT_DIR = Path(f'/home/ubuntu/efs/repositories/PCDet')