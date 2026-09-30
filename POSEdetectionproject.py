import os
import cv2
import gradio as gr
import requests
import tempfile
import shutil
import yt_dlp

from urllib.parse import urlparse
from ultralytics import YOLO


# ============================================================
# 1. LOAD YOLO MODEL
# ============================================================

MODEL_PATH = "yolo26n-pose.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' was not found.\n"
        "Make sure yolo26n-pose.pt is in the same folder as this Python file."
    )

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully!")


# ============================================================
# 2. DOWNLOAD VIDEO FROM URL
# ============================================================

def download_video_from_url(url):

    url = url.strip()

    if not url:
        raise Exception("Video URL is empty.")

    print("\nVideo URL received:")
    print(url)

    # Create temporary folder
    download_folder = tempfile.mkdtemp()

    # --------------------------------------------------------
    # FIRST: TRY DIRECT VIDEO DOWNLOAD
    # --------------------------------------------------------

    try:

        print("Trying direct video download...")

        response = requests.get(
            url,
            stream=True,
            timeout=60,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "content-type",
            ""
        ).lower()

        parsed_url = urlparse(url)

        url_path = parsed_url.path.lower()

        video_extensions = (
            ".mp4",
            ".webm",
            ".mov",
            ".avi",
            ".mkv",
            ".m4v"
        )

        is_video = (
            "video/" in content_type
            or url_path.endswith(video_extensions)
        )

        if is_video:

            output_path = os.path.join(
                download_folder,
                "input_video.mp4"
            )

            print("Direct video detected.")
            print("Downloading...")

            with open(output_path, "wb") as file:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:
                        file.write(chunk)

            # Check file size
            file_size = os.path.getsize(
                output_path
            )

            if file_size > 0:

                print(
                    f"Video downloaded successfully: "
                    f"{file_size / (1024 * 1024):.2f} MB"
                )

                return output_path, download_folder

    except Exception as e:

        print(
            "Direct download failed."
        )

        print(
            "Trying yt-dlp..."
        )

    # --------------------------------------------------------
    # SECOND: TRY YT-DLP
    # --------------------------------------------------------

    try:

        print("Using yt-dlp to download video...")

        output_template = os.path.join(
            download_folder,
            "input_video.%(ext)s"
        )

        ydl_options = {

            # Prefer MP4 when available.
            # Maximum 720p to reduce server load.
            "format":
                "best[ext=mp4][height<=720]/"
                "best[height<=720]/"
                "best",

            "outtmpl": output_template,

            "noplaylist": True,

            "quiet": False,

            "no_warnings": False,

            "retries": 3,

            "socket_timeout": 60,

            "restrictfilenames": True,
        }

        with yt_dlp.YoutubeDL(
            ydl_options
        ) as ydl:

            info = ydl.extract_info(
                url,
                download=True
            )

            downloaded_file = ydl.prepare_filename(
                info
            )

        # ----------------------------------------------------
        # FIND DOWNLOADED FILE
        # ----------------------------------------------------

        if os.path.exists(downloaded_file):

            print(
                "Video downloaded successfully."
            )

            return (
                downloaded_file,
                download_folder
            )

        # Sometimes extension changes after download.
        # Search the temporary folder.
        for filename in os.listdir(
            download_folder
        ):

            full_path = os.path.join(
                download_folder,
                filename
            )

            if os.path.isfile(full_path):

                print(
                    "Downloaded video found:"
                )

                print(full_path)

                return (
                    full_path,
                    download_folder
                )

        raise Exception(
            "yt-dlp finished but the video file "
            "could not be found."
        )

    except Exception as e:

        # Remove temporary folder
        shutil.rmtree(
            download_folder,
            ignore_errors=True
        )

        raise Exception(
            "Could not download the video from this URL.\n\n"
            f"Reason: {str(e)}"
        )


# ============================================================
# 3. PROCESS VIDEO USING YOLO
# ============================================================

def process_video(
    uploaded_video,
    video_url
):

    download_folder = None
    output_path = None

    try:

        # ====================================================
        # SELECT VIDEO SOURCE
        # ====================================================

        if uploaded_video:

            print("\nUsing uploaded/webcam video.")

            input_video = uploaded_video

        elif video_url and video_url.strip():

            print("\nUsing video URL.")

            input_video, download_folder = (
                download_video_from_url(
                    video_url
                )
            )

        else:

            return (
                None,
                "❌ Please upload/record a video OR enter a video URL."
            )

        print("\nInput video:")
        print(input_video)

        # ====================================================
        # OPEN VIDEO
        # ====================================================

        cap = cv2.VideoCapture(
            input_video
        )

        if not cap.isOpened():

            return (
                None,
                "❌ Could not open the video."
            )

        # ====================================================
        # GET VIDEO INFORMATION
        # ====================================================

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        # Some videos don't provide FPS.
        if fps <= 0:

            fps = 25

        print("\nVideo information:")
        print(
            f"Width: {width}"
        )
        print(
            f"Height: {height}"
        )
        print(
            f"FPS: {fps}"
        )
        print(
            f"Total frames: {total_frames}"
        )

        # ====================================================
        # CREATE OUTPUT FILE
        # ====================================================

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_path = output_file.name

        output_file.close()

        # ====================================================
        # CREATE VIDEO WRITER
        # ====================================================

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():

            cap.release()

            return (
                None,
                "❌ Could not create output video."
            )

        # ====================================================
        # YOLO PROCESSING
        # ====================================================

        print("\nStarting YOLO processing...")

        processed_frames = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # ------------------------------------------------
            # YOLO INFERENCE
            # ------------------------------------------------

            results = model(
                frame,
                verbose=False
            )

            # ------------------------------------------------
            # DRAW YOLO RESULTS
            # ------------------------------------------------

            annotated_frame = results[
                0
            ].plot()

            # ------------------------------------------------
            # SAVE FRAME
            # ------------------------------------------------

            writer.write(
                annotated_frame
            )

            processed_frames += 1

            # Print progress every 50 frames
            if processed_frames % 50 == 0:

                print(
                    f"Processed "
                    f"{processed_frames} frames..."
                )

        # ====================================================
        # RELEASE VIDEO
        # ====================================================

        cap.release()

        writer.release()

        # ====================================================
        # CHECK RESULT
        # ====================================================

        if processed_frames == 0:

            if output_path and os.path.exists(
                output_path
            ):

                os.remove(
                    output_path
                )

            return (
                None,
                "❌ No frames were processed."
            )

        print("\nProcessing completed!")

        print(
            f"Total processed frames: "
            f"{processed_frames}"
        )

        print(
            f"Output video: "
            f"{output_path}"
        )

        # ====================================================
        # RETURN RESULT
        # ====================================================

        return (
            output_path,
            f"✅ Processing completed successfully!\n"
            f"Frames processed: {processed_frames}"
        )

    except Exception as e:

        print("\nERROR:")
        print(str(e))

        return (
            None,
            f"❌ Error:\n{str(e)}"
        )

    finally:

        # ====================================================
        # CLEAN DOWNLOADED VIDEO
        # ====================================================

        if download_folder:

            try:

                shutil.rmtree(
                    download_folder,
                    ignore_errors=True
                )

                print(
                    "Temporary downloaded video deleted."
                )

            except Exception:

                pass


# ============================================================
# 4. GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="YOLO Pose Detection"
) as demo:

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    gr.Markdown(
        """
        # 🤖 YOLO Pose Detection

        Upload a video, record a video using your webcam,
        or provide a supported video URL.

        The YOLO pose model will detect and annotate
        objects/poses in the video.
        """
    )

    # --------------------------------------------------------
    # INPUT / OUTPUT
    # --------------------------------------------------------

    with gr.Row():

        # ====================================================
        # LEFT SIDE
        # ====================================================

        with gr.Column():

            gr.Markdown(
                "### 📥 Input Video"
            )

            # Upload OR webcam
            video_input = gr.Video(
                label="📹 Upload or Record Video",
                sources=[
                    "upload",
                    "webcam"
                ]
            )

            gr.Markdown(
                "### 🔗 OR use a Video URL"
            )

            video_url = gr.Textbox(
                label="Video URL",
                placeholder=(
                    "Paste a supported video URL here..."
                )
            )

            process_button = gr.Button(
                "🚀 Detect Pose",
                variant="primary"
            )

        # ====================================================
        # RIGHT SIDE
        # ====================================================

        with gr.Column():

            gr.Markdown(
                "### 🎯 Detection Result"
            )

            video_output = gr.Video(
                label="YOLO Detection Result",
                format="mp4"
            )

            status_output = gr.Textbox(
                label="Status",
                lines=4
            )

    # --------------------------------------------------------
    # BUTTON ACTION
    # --------------------------------------------------------

    process_button.click(
        fn=process_video,
        inputs=[
            video_input,
            video_url
        ],
        outputs=[
            video_output,
            status_output
        ]
    )

    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    gr.Markdown(
        """
        ### 📌 How to use

        **Option 1 — Upload**

        Select a video from your computer.

        **Option 2 — Webcam**

        Record a video using your webcam.

        **Option 3 — Video URL**

        Paste a supported video URL.

        **Note:** Not every website URL is downloadable.
        Private, restricted, or unsupported videos may fail.

        ### ⚡ Tip

        Short videos are recommended for the demo because
        YOLO has to process every frame.
        """
    )


# ============================================================
# 5. START APPLICATION
# ============================================================

if __name__ == "__main__":

    print(
        "\n================================================"
    )

    print(
        "YOLO POSE DETECTION WEB APPLICATION"
    )

    print(
        "================================================"
    )

    print(
        "Starting Gradio..."
    )

    demo.launch()