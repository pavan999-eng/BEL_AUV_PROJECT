import os
import cv2
import joblib
import numpy as np

from PIL import Image

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from feature_extractor import extract_hog_features

# ----------------------------------------
# PATHS
# ----------------------------------------

IMAGE_PATH = "../phase2_cnn/dataset/images/train"

LABEL_PATH = "../phase2_cnn/dataset/labels/train"

# ----------------------------------------
# CLASS NAMES
# ----------------------------------------


CLASS_NAMES = {
    0: "ball",
    1: "plane",
    2: "cylinder",
    3: "cube",
    4: "tyre",
    5: "circle cage",
    6: "human body",
    7: "metal bucket",
    8: "rov",
    9: "square cage"
}

# ----------------------------------------
# DATA STORAGE
# ----------------------------------------

X = []
y = []

# ----------------------------------------
# LOOP THROUGH IMAGES
# ----------------------------------------

for image_name in os.listdir(IMAGE_PATH):

    if not image_name.endswith((".jpg", ".png", ".bmp")):
        continue

    image_path = os.path.join(
        IMAGE_PATH,
        image_name
    )

    label_name = os.path.splitext(image_name)[0] + ".txt"

    label_path = os.path.join(
        LABEL_PATH,
        label_name
    )

    # Skip if no label file

    if not os.path.exists(label_path):
        continue

    # Load image

    image = cv2.imread(image_path)

    if image is None:
        continue

    image_h, image_w = image.shape[:2]

    # Read YOLO labels

    with open(label_path, "r") as f:

        lines = f.readlines()

    # Process each object

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # Convert YOLO -> pixel coords

        x1 = int((x_center - width / 2) * image_w)
        y1 = int((y_center - height / 2) * image_h)

        x2 = int((x_center + width / 2) * image_w)
        y2 = int((y_center + height / 2) * image_h)

        # Crop object

        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            continue

        # Convert BGR -> RGB

        cropped = cv2.cvtColor(
            cropped,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(cropped)

        # Extract HOG

        features = extract_hog_features(
            pil_image
        )

        X.append(features)

        y.append(
            CLASS_NAMES[class_id]
        )

# ----------------------------------------
# NUMPY
# ----------------------------------------

X = np.array(X)
y = np.array(y)

print(f"\nTotal Samples: {len(X)}")

# ----------------------------------------
# TRAIN TEST SPLIT
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------------
# SVM MODEL
# ----------------------------------------

svm_model = SVC(
    kernel='rbf',
    probability=True
)

print("\nTraining SVM...")

svm_model.fit(
    X_train,
    y_train
)

# ----------------------------------------
# PREDICTION
# ----------------------------------------

predictions = svm_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# ----------------------------------------
# SAVE MODEL
# ----------------------------------------

joblib.dump(
    svm_model,
    "../models/image_svm.pkl"
)

print("\nImage SVM saved successfully.")