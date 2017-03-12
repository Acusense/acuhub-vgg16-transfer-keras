import os, json

config_path = os.path.join(os.environ['BASE_PATH'], 'config.json')
config_dict = json.load(open(config_path))
general_dict = config_dict["general"]
visualizer_dict = config_dict["visualizer"]