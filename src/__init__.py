import os, json

config_path = os.path.join(os.environ['INPUT_DIR'], 'config.json')
config_dict = json.load(open(config_path))
general_dict = config_dict["general"]
data_path = os.environ['DATA_DIR']
w