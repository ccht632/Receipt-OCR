import time

import streamlit as st
from PIL import Image
from preprocess import preprocess_image

st.set_page_config(
    page_title="AI Owe You",
    page_icon="🧾",
    layout="wide"
)

# page title
st.title("AI Owe You")
st.caption("Smart Receipt Splitter using AI")

st.write(
    "Upload a receipt and let AI automatically extract items for bill splitting."
)

st.divider()

# receipt upload
st.header("📤 Upload Receipt")

uploaded_file = st.file_uploader(
    "Choose a receipt image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # yolo_input  -> small 640x640 letterboxed array, only for the YOLO model
    # preview_image -> full-resolution clean image, use this for display / OCR
    yolo_input, preview_image = preprocess_image(image)

    left, right = st.columns([2, 1])

    with left:

        st.subheader("🖼 Receipt Preview")

        st.image(
            preview_image,
            caption="Processed Receipt",
            use_container_width=True,
            clamp=True
        )

    with right:

        st.subheader("⚙ AI Settings")

        algorithm = st.selectbox(
            "Detection Algorithm",
            [
                "YOLO",
                "CNN (sliding window)",
                "Mask R-CNN"
            ]
        )

        process = st.button(
            "🚀 Start Processing",
            use_container_width=True
        )

    # image processing
    if process:

        st.divider()

        st.subheader("🤖 AI Processing")

        progress = st.progress(0)

        status = st.empty()

        # NOTE: placeholder progress bar for UI layout only — steps are not
        # tied to real processing yet. Replace with actual timings once
        # YOLO inference + OCR are wired in (e.g. update progress after each
        # real step finishes instead of using time.sleep()).
        steps = [
            "Uploading receipt...",
            "Image preprocessing...",
            "Running OCR...",
            "Extracting items...",
            "Finished!"
        ]

        for i in range(5):

            status.info(steps[i])

            progress.progress((i + 1) * 20)

            time.sleep(0.8)

        status.success("Processing Complete ✅")

        st.divider()

        st.header("📋 Extracted Items")
        st.info("OCR results will appear here.")

        st.divider()

        st.header("👥 People")
        st.info("Add people here.")

        st.divider()

        st.header("💰 Bill Summary")
        st.info("Bill calculation will appear here.")

        st.divider()

        st.header("📊 Evaluation")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", "--")
        col2.metric("Confidence", "--")
        col3.metric("Processing Time", "--")
        col4.metric("Items", "--")