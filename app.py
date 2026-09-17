import streamlit as st
import cv2
import tempfile
import os
from PIL import Image
from ultralytics import YOLO

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🎯"
)

st.title("🎯 Object Detection & Tracking")
st.write("Upload an image or video to detect and track objects.")

# -----------------------------
# Load YOLO model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------
# Upload image or video
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:

    file_name = uploaded_file.name.lower()

    # ==================================================
    # IMAGE
    # ==================================================
    if file_name.endswith((".jpg", ".jpeg", ".png")):

        image = Image.open(uploaded_file)

        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

        if st.button("🔍 Detect Objects"):

            with st.spinner("Detecting objects..."):

                results = model.predict(
                    image,
                    conf=0.25,
                    verbose=False
                )

                detected_image = results[0].plot()

            st.subheader("🎯 Detected Image")

            st.image(
                detected_image,
                channels="BGR",
                use_container_width=True
            )

            st.success("✅ Object detection completed!")

    # ==================================================
    # VIDEO
    # ==================================================
    elif file_name.endswith((".mp4", ".avi", ".mov", ".mkv")):

        # Save uploaded video
        input_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_path = input_temp.name

        input_temp.write(uploaded_file.getbuffer())
        input_temp.close()

        st.subheader("🎥 Original Video")
        st.video(input_path)

        if st.button("▶️ Start Detection & Tracking"):

            with st.spinner(
                "🔄 Processing video... Please wait."
            ):

                cap = cv2.VideoCapture(input_path)

                if not cap.isOpened():
                    st.error("❌ Could not open the video.")
                    st.stop()

                width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )

                height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                fps = cap.get(
                    cv2.CAP_PROP_FPS
                )

                if fps <= 0:
                    fps = 30

                # Output file
                output_path = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ).name

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
                    cap.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )

                progress = st.progress(0)

                frame_number = 0

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    # Use ByteTrack
                    results = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=0.25,
                        verbose=False
                    )

                    # Draw boxes and tracking IDs
                    annotated_frame = results[0].plot()

                    out.write(annotated_frame)

                    frame_number += 1

                    if total_frames > 0:

                        percentage = (
                            frame_number / total_frames
                        )

                        progress.progress(
                            min(percentage, 1.0)
                        )

                cap.release()
                out.release()

                progress.progress(1.0)

            st.success(
                "✅ Detection and tracking completed!"
            )

            st.subheader("🎯 Processed Video")

            st.video(output_path)

            # Download button
            with open(output_path, "rb") as video:

                st.download_button(
                    label="⬇️ Download Output Video",
                    data=video,
                    file_name="object_detection_tracking.mp4",
                    mime="video/mp4"
                )

            # Remove temporary input
            if os.path.exists(input_path):
                os.remove(input_path)

    else:

        st.error(
            "❌ Please upload a JPG, PNG, MP4, AVI, MOV, or MKV file."
        )