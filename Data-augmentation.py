import albumentations as A
from PIL import Image
import numpy as np
import cv2

# Create a pipline with 4 different transformations. 
transform = A.Compose(
    [
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=.5, contrast_limit=.3),
        A.Rotate(),
    ]
)

# Read the image and convert it to a numpy array
pillow_image = Image.open("/content/drive/MyDrive/images/images-uploads-clownfish_400_q85.jpg")
image = np.array(pillow_image)

# Apply transformation
transformed = transform(image=image)

# Access and show transformation
transformed_image = transformed["image"]
img = Image.fromarray(transformed_image)

img.show()
