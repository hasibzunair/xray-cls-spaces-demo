import numpy as np
import cv2

from tf_explain.core.grad_cam import GradCAM
from PIL import Image

IMAGE_SIZE = 224

def preprocess_image(uploaded_file):
    # Load image
    img_array = np.array(Image.open(uploaded_file))
    # Normalize to [0,1]
    img_array = img_array.astype('float32')
    img_array /= 255
    # Check that images are 2D arrays
    if len(img_array.shape) > 2:
        img_array = img_array[:, :, 0]
    # Convert to 3-channel
    img_array = np.stack((img_array, img_array, img_array), axis=-1)
    # Convert to array
    img_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    return img_array


def get_gradcam(uploaded_file, model, layer_name, predictions):
    # Load and process image
    image = preprocess_image(uploaded_file)
    # Add batch axis
    image = np.expand_dims(image, axis=0)
    y_classes = predictions.argmax(axis=-1)
    # GradCAM
    explainer = GradCAM()
    output = explainer.explain(
        validation_data=(image, None),
        model=model,
        layer_name=layer_name,
        class_index=y_classes[0],
        colormap=cv2.COLORMAP_TURBO,
    )
    return output

