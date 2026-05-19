# app.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="Underwater Sonar Detection",
    layout="wide"
)

# -----------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------

MODEL_PATH = os.path.join(
    "phase2_cnn",
    "runs",
    "detect",
    "train-2",
    "weights",
    "best.pt"
)

model = YOLO(MODEL_PATH)

# -----------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------

st.markdown("""
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

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #f2f0ec;
    border-right: 1px solid #e6e6e6;
}

section[data-testid="stSidebar"] * {
    color: #111111 !important;
}

/* Hero */

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

/* Stat cards */

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

/* Upload */

[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #dcdcdc;
    border-radius: 22px;
    padding: 18px;
}

/* Images */

img {
    border-radius: 18px;
}

/* Detection cards */

.detect-card {
    background: white;
    border: 1px solid #ebebeb;
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 18px;
}

.stProgress > div > div > div > div {
    background: #111111;
}

.stSuccess {
    background: #eaf7ec;
    border: 1px solid #cae7d1;
    color: #1c5c2c;
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

st.sidebar.markdown("""
## Underwater
## Sonar Detection
""")

st.sidebar.markdown("---")

st.sidebar.markdown("### Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.5
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### System Information

- YOLOv8 Detection Engine
- Sonar Object Recognition
- Real-time Multi-class Detection
- BEL Internship Project
""")

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
        Professional underwater object detection system powered by YOLOv8 and sonar intelligence for real-time multi-class analysis.
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------
# TOP STATS
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
            <div class="metric-value" style="color:#15803d;">
                Active
            </div>
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

    # Save temp image

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".bmp"
    )

    image.save(temp_file.name)

    # Run YOLO

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
            use_container_width=True
        )

    with col2:

        st.markdown("### Detection Output")

        st.image(
            result_image,
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)


    # -------------------------------------------------
    # # DETECTED OBJECTS
    # # -------------------------------------------------
    st.markdown("## Detected Objects")
    boxes = results[0].boxes
    if len(boxes) == 0:
        st.warning("No objects detected.")
    else:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            percent = conf * 100
            card_html = f"""
            <div class="detect-card">
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                margin-bottom:14px;
                ">
                <div>
                <div style="
                        font-size:30px;
                        font-weight:800;
                        color:#111111;
                          ">
                          {class_name}
                          </div><div style="
                          color:#666666;
                          margin-top:6px;
                          font-size:16px;
                          ">
                          Confidence Score: {conf:.2f}
                          </div>
                          </div>
                          <div style="
                          background:#111111;
                          color:white;
                          padding:10px 14px;
                          border-radius:12px;
                          font-weight:700;
                          font-size:14px;
                          ">
                          {percent:.2f}%
                          </div>
                          </div>
                          </div>
                          """
            st.markdown(
            card_html,
            unsafe_allow_html=True
        )

        st.progress(conf)
        st.success(
        f"{len(boxes)} object(s) detected successfully"
    )