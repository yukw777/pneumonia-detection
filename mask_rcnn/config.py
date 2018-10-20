import configparser

from ast import literal_eval

from mrcnn.config import Config


def from_config_file(config_file):
    c = Config()
    config = configparser.ConfigParser()
    config.read(config_file)
    for k, v in config['mask_rcnn'].items():
        setattr(c, k.upper(), literal_eval(v))
    c.__init__()
    return c


if __name__ == '__main__':
    import sys
    print(from_config_file(sys.argv[1]).display())
