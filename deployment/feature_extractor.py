from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np

def extract_hog_features(image):

    # PIL -> numpy
    image = np.array(image)

    # grayscale
    gray = rgb2gray(image)

    # resize
    resized = resize(gray, (128, 128))

    # HOG extraction
    features = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    return features