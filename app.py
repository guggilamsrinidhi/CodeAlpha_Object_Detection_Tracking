import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🎯"
)

st.title("🎯 Object Detection & Tracking")
st.write("Upload a video to detect and track objects.")

uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:

    input_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name

    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(input_path)

    if st.button("▶️ Start Detection"):

        st.info("Processing video... Please wait.")

        model = YOLO("yolov8n.pt")

        cap = cv2.VideoCapture(input_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0:
            fps = 30

        output_path = "output.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                verbose=False
            )

            annotated_frame = results[0].plot()

            out.write(annotated_frame)

        cap.release()
        out.release()

        st.success("✅ Detection and tracking completed!")

        st.video(output_path)

        with open(output_path, "rb") as video_file:
            st.download_button(
                label="⬇️ Download Output Video",
                data=video_file,
                file_name="object_detection_tracking.mp4",
                mime="video/mp4"
            )

        os.remove(input_path)