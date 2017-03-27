from model import model
import numpy as np
from PIL import Image


def test(image_path, snapshot_path, transform=False):

    img = Image.open(image_path) # WARNING : this image is well centered and square
    img = img.resize(model.inputs[0].shape)

    imarr = np.array(img).astype(np.float32)

    imarr = imarr.transpose((2, 0, 1))
    imarr = np.expand_dims(imarr, axis=0)

    model.load_weights(snapshot_path)

    out = model.predict(imarr)

    best_index = np.argmax(out, axis=1)[0]
    print best_index
