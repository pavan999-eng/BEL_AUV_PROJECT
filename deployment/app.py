import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import streamlit.components.v1 as components

import sys
import os
import joblib

sys.path.append(
    os.path.abspath("..")
)

from feature_extractor import (
    extract_hog_features
)
 


# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="Underwater Sonar Detection",
    layout="wide"
)

import gdown

# -----------------------------------------------------
# BASE DIRECTORY
# -----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "..",
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------------------------
# YOLO MODEL PATH
# -----------------------------------------------------

YOLO_PATH = os.path.join(
    MODEL_DIR,
    "best.pt"
)

# -----------------------------------------------------
# SVM MODEL PATH
# -----------------------------------------------------

SVM_PATH = os.path.join(
    MODEL_DIR,
    "image_svm.pkl"
)

# -----------------------------------------------------
# DOWNLOAD LARGE SVM MODEL
# -----------------------------------------------------

if not os.path.exists(SVM_PATH):

    url = "https://drive.google.com/uc?id=1fmVyVRQJAp3IrZ9XY-gVYmRAkbgPMRQn"

    gdown.download(
        url,
        SVM_PATH,
        quiet=False
    )

# -----------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------

model = YOLO(YOLO_PATH)

svm_model = joblib.load(SVM_PATH)
# -----------------------------------------------------
# CSS
# -----------------------------------------------------

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background: #f7f5f2;
        color: #111111;
    }

    .stApp {
        background: #f7f5f2;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1450px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    section[data-testid="stSidebar"] {
        background: #f2f0ec;
        border-right: 1px solid #e6e6e6;
    }

    section[data-testid="stSidebar"] * {
        color: #111111 !important;
    }

    .hero-title {
        font-size: 58px;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 12px;
        color: #111111;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #666666;
        line-height: 1.7;
        max-width: 760px;
        margin-bottom: 34px;
    }

    .metric-card {
        background: white;
        border: 1px solid #ebebeb;
        border-radius: 22px;
        padding: 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }

    .metric-title {
        color: #666666;
        font-size: 14px;
        margin-bottom: 18px;
    }

    .metric-value {
        font-size: 42px;
        font-weight: 800;
        color: #111111;
        margin-bottom: 10px;
    }

    .metric-sub {
        color: #777777;
        font-size: 15px;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #dcdcdc;
        border-radius: 22px;
        padding: 18px;
    }

    img {
        border-radius: 18px;
    }

    .stSuccess {
        background: #eaf7ec;
        border: 1px solid #cae7d1;
        color: #1c5c2c;
        border-radius: 14px;
    }

    .stExpander {
    margin-top: -10px;
    margin-bottom: 10px;
    }

    .streamlit-expanderHeader {
    font-size: 14px !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    }
    
    .streamlit-expanderContent {
    background: white;
    border-radius: 12px;
    padding: 10px;
    }
    .stExpander {
    margin-top: -8px;
    margin-bottom: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

st.sidebar.markdown("## Underwater\n## Sonar Detection")

st.sidebar.markdown("---")

st.sidebar.markdown("### Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.5
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### System Information

    - YOLOv8 Detection Engine
    - Sonar Object Recognition
    - Real-time Multi-class Detection
    - BEL Internship Project
    """
)

# -----------------------------------------------------
# HERO
# -----------------------------------------------------

st.markdown(
    """
    <div class="hero-title">
        Underwater Sonar Detection
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-subtitle">
        Professional underwater object detection system powered by YOLOv8 and sonar intelligence.
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------
# STATS
# -----------------------------------------------------

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Total Classes</div>
            <div class="metric-value">10</div>
            <div class="metric-sub">Object Categories</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Detection Model</div>
            <div class="metric-value">YOLOv8</div>
            <div class="metric-sub">Latest Architecture</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">System Status</div>
            <div class="metric-value" style="color:#15803d;;">Active</div>
            <div class="metric-sub">All Systems Operational</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------
# FILE UPLOADER
# -----------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Sonar Image",
    type=["jpg", "png", "bmp"]
)

# -----------------------------------------------------
# PROCESS IMAGE
# -----------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".bmp"
    )

    image.save(temp_file.name)

    with st.spinner("Running Detection Model..."):

        results = model.predict(
            temp_file.name,
            conf=confidence
        )

    result_image = results[0].plot()

    # -------------------------------------------------
    # IMAGE DISPLAY
    # -------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Input Image")

        st.image(
            image,
            width='stretch'
        )

    with col2:

        st.markdown("### Detection Output")

        st.image(
            result_image,
            width='stretch'
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------
    # DETECTED OBJECTS
    # -------------------------------------------------

    st.markdown(
    """
    <div style="
        font-size:34px;
        font-weight:800;
        color:#111111;
        margin-top:10px;
        margin-bottom:18px;
    ">
        Detected Objects
    </div>
    """,
    unsafe_allow_html=True
    )

    boxes = results[0].boxes

    if len(boxes) == 0:

        st.warning("No objects detected.")

    else:

        for box in boxes:

            cls_id = int(box.cls[0])

            conf = float(box.conf[0])

            class_name = model.names[cls_id]

            percent = conf * 100

            # Bounding Box

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Crop Object

            cropped_object = image.crop(
                (x1, y1, x2, y2)
            )

            # -------------------------------------
            # HOG FEATURE EXTRACTION
            # -------------------------------------

            features = extract_hog_features(
                 cropped_object
            )

            # -------------------------------------
            #  SVM PREDICTION
            # -------------------------------------
            
            prediction = svm_model.predict(
                [features]
            )
            
            svm_prediction = prediction[0]
            # -------------------------------------
            #  FINAL SVM CLASS
            # -------------------------------------

            svm_class = str(svm_prediction)

            # -------------------------------------
            # HYBRID DECISION LOGIC
            # -------------------------------------
            if conf < 0.60:
                refined_prediction = svm_class
            else:
                refined_prediction = class_name


            # -------------------------------------------------
            # CARD HTML
            # -------------------------------------------------

            card_html = f"""
            <div style="
                background:white;
                border:1px solid #ebebeb;
                border-radius:20px;
                padding:24px;
                margin-bottom:20px;
                font-family:Inter,sans-serif;
                box-shadow:0 2px 10px rgba(0,0,0,0.03);
            ">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    margin-bottom:20px;
                ">

                    <div>

                        <div style="
                            font-size:28px;
                            font-weight:800;
                            color:#111111;
                            line-height:1;
                        ">
                            {class_name}
                        </div>

                        <div style="
                            color:#666666;
                            margin-top:8px;
                            font-size:15px;
                        ">
                            YOLO Detection
                        </div>

                    </div>

                    <div style="
                        background:#111111;
                        color:white;
                        padding:10px 16px;
                        border-radius:12px;
                        font-weight:700;
                        font-size:14px;
                    ">
                        {percent:.2f}%
                    </div>

                </div>

                <div style="
                    width:100%;
                    height:10px;
                    background:#eeeeee;
                    border-radius:999px;
                    overflow:hidden;
                    margin-bottom:20px;
                ">

                    <div style="
                        width:{percent}%;
                        height:100%;
                        background:#111111;
                        border-radius:999px;
                    ">
                    </div>

                </div>

                <div style="
                    background:#f7f7f7;
                    border-radius:14px;
                    padding:16px;
                ">

                    <div style="
                        color:#777777;
                        font-size:13px;
                        margin-bottom:8px;
                    ">
                        Refined Classification
                    </div>

                    <div style="
                        font-size:22px;
                        font-weight:700;
                        color:#111111;
                    ">
                        {refined_prediction}
                    </div>

                </div>

            </div>
            """

            components.html(
                card_html,
                height=220
            )

            # -------------------------------------------------
            # CROPPED IMAGE
            # -------------------------------------------------

            with st.expander(
                f"View Cropped Object - {class_name}",
                expanded=False
            ):

                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    st.image(
                        cropped_object,
                        width=140
                    )

        # -------------------------------------------------
        # SUCCESS MESSAGE
        # -------------------------------------------------

        st.success(
            f"{len(boxes)} object(s) detected successfully"
        )