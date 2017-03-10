import argparse
from training import train
from visualization import activation, filter, filter_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("--layer-name", dest="layer_name", default="block_conv4",
                        help="name of the layer")
    parser.add_argument("--image-path", dest="image_path", default="/training_files/data/dog1.jpg")
    args = parser.parse_args()
    if args.mode == "training":
        train.train()
    elif args.mode == "filter_vis":
        filter.filter_vis(args.layer_name)
    elif args.mode == "filter_vis_image":
        filter_image.filter_on_image(args.image_path)
    elif args.mode == "activation_map_image":
        # Default "predictions"
        activation.activation_vis(args.layer_name, args.image_path)
