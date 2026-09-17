import streamlit as st
import cv2
import tempfile
import os
from PIL import Image
from ultralytics import YOLO

# Page configuration
st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🎯",
    layout="centered"
)

# Title
st.title("🎯 Object Detection & Tracking")
st.write("Upload an image or video to detect and track objects.")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=[
        "jpg",
        "jpeg",
        "png",
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)

if uploaded_file is not None:

    # Get file type
    file_type = uploaded_file.type

    # =========================
    # IMAGE DETECTION
    # =========================
    if file_type.startswith("image"):

        image = Image.open(uploaded_file)

        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

        if st.button("🔍 Detect Objects"):

            with st.spinner("Detecting objects..."):

                results = model.predict(
                    source=image,
                    conf=0.25,
                    verbose=False
                )

                detected_image = results[0].plot()

            st.subheader("🎯 Detected Objects")
            st.image(
                detected_image,
                channels="BGR",
                use_container_width=True
            )

            st.success("✅ Object detection completed!")

    # =========================
    # VIDEO DETECTION & TRACKING
    # =========================
    elif file_type.startswith("video"):

        # Save uploaded video temporarily
        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_path = input_file.name

        input_file.write(uploaded_file.getbuffer())
        input_file.close()

        st.subheader("🎥 Original Video")
        st.video(input_path)

        if st.button("▶️ Start Detection & Tracking"):

            output_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ).name

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                st.error("❌ Could not open the uploaded video.")
                os.remove(input_path)
                st.stop()

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 30

            # MP4 output
            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            progress = st.progress(0)

            with st.spinner(
                "🔄 Detecting and tracking objects..."
            ):

                frame_count = 0

                while cap.isOpened():

                    ret, frame = cap.read()

                    if not ret:
                        break

                    # YOLO tracking
                    results = model.track(
                        frame,
                        persist=True,
                        conf=0.25,
                        verbose=False
                    )

                    # Draw detections
                    annotated_frame = results[0].plot()

                    # Write frame
                    out.write(annotated_frame)

                    frame_count += 1

                    if total_frames > 0:
                        progress.progress(
                            min(
                                frame_count / total_frames,
                                1.0
                            )
                        )

            cap.release()
            out.release()

            progress.progress(1.0)

            st.success(
                "✅ Object detection and tracking completed!"
            )

            # Show result
            st.subheader("🎯 Processed Video")
            st.video(output_path)

            # Download result
            with open(output_path, "rb") as video_file:

                st.download_button(
                    label="⬇️ Download Output Video",
                    data=video_file,
                    file_name="object_detection_tracking.mp4",
                    mime="video/mp4"
                )

            # Remove input temporary file
            if os.path.exists(input_path):
                os.remove(input_path)