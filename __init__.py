__project__ = 'acuhub'
__version__ = '0.0.0'

VERSION = "{0} v{1}".format(__project__, __version__)
import os, json

config_path = os.path.join(os.environ['BASE_PATH'], 'config.json')
config_dict = json.load(open(config_path))
general_dict = config_dict["general"]

data_path = os.path.join(os.environ['BASE_PATH'], 'data')
