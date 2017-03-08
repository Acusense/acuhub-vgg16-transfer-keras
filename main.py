import argparse
from training import train
from visualization import activation, filter, filter_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    args = parser.parse_args()
    if args.mode == "training":
        train.train()
    elif args.mode == "visualization":
        filter.filter_vis('block_conv4')
        activation.activation_vis("predictions")
        filter_image.activation_map("test.jpg", "maps")
