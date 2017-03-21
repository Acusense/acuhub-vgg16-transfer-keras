import argparse
from training import train
from visualization import activation_image, filter, filter_image, model
from __init__ import config_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    args = parser.parse_args()
    if args.command == "train":
        train.train()
    elif args.command == "model":
        model.save(config_dict["visualization"]["model"]["type"])
    elif args.command == "filter":
        filter.filter_vis(config_dict["visualization"]["filter"]["layer_name"],
                          config_dict["visualization"]["filter"]["image_path"])
    elif args.command == "filter_image":
        filter_image.filter_on_image(config_dict["visualization"]["filter_image"]["image_path"])
    elif args.command == "activation_image":
        activation_image.activation_vis(config_dict["visualization"]["activation_image"]["layer_name"],
                                        config_dict["visualization"]["activation_image"]["image_path"])
