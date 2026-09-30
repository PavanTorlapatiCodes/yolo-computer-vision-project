# 🚀 YOLO Computer Vision – Multiple Tasks & Real-World Video AI Application

A hands-on **Computer Vision project using YOLO, Python, and OpenCV**, covering multiple real-world computer vision tasks such as **Object Detection, Object Counting, Object Tracking, Object Segmentation, Person Detection, Pose Detection, and Semantic Segmentation**.

The project was developed by experimenting with different YOLO model versions, including **YOLOv8, YOLO11, and YOLO26**, and finally converting the pose-detection workflow into an interactive web application where users can upload videos, record videos using their webcam, or provide a supported video URL and download the processed result.

---

## 📌 Project Overview

This project started as a practical exploration of different **YOLO-based Computer Vision tasks**.

I implemented and tested:

- Object Detection
- Object Counting
- Object Tracking
- Object Segmentation
- Person Detection
- Pose Detection
- Semantic Segmentation

After successfully implementing these tasks, I extended the project into a **web-based YOLO Pose Detection application**.

The final application allows users to:

1. Upload a video
2. Record a video using their webcam
3. Provide a supported video URL
4. Run YOLO Pose Detection
5. View the processed video
6. Download the annotated output video

---

# 🎯 Objectives

The main objectives of this project were:

- Understand modern YOLO-based computer vision models.
- Implement different computer vision tasks using Python.
- Work with image and video data.
- Understand object detection and segmentation concepts.
- Explore human pose estimation.
- Process video frames using OpenCV.
- Understand how different YOLO tasks are applied in real-world scenarios.
- Convert a Python computer vision script into an interactive application.
- Allow users to interact with the AI model through a web interface.

---

# 🤖 YOLO Tasks Implemented

## 1. 🔍 Object Detection

Object Detection identifies objects present in an image or video and draws bounding boxes around them.

### Example output

```text
Car
Person
Dog
Bottle
Chair

The model provides:

Object class
Bounding box
Confidence score
Technologies
Python
YOLO
OpenCV

2. 🔢 Object Counting

Object Counting extends object detection by determining how many objects are present.

For example:

People detected: 5
Cars detected: 3
Buses detected: 2

This can be useful for:

Crowd monitoring
Traffic analysis
Retail analytics
Vehicle counting
People counting

3. 🎯 Object Tracking

Object Tracking follows detected objects across multiple video frames.

Instead of only detecting:

Car

the tracker can maintain an identity such as:

Car → ID 3

This allows the system to understand that the same object is moving across different frames.

Applications
Traffic monitoring
Vehicle tracking
People tracking
Surveillance
Sports analysis

4. 🟢 Object Segmentation

Object Segmentation identifies the exact pixels belonging to an object instead of only drawing a rectangular bounding box.

For example:

Image
   ↓
YOLO Segmentation
   ↓
Object Mask

This provides more detailed information about the shape and location of objects.

Applications
Medical image analysis
Autonomous systems
Industrial inspection
Background/object analysis

5. 👤 Person Detection

Person Detection focuses specifically on identifying people in images or videos.

The model detects humans and provides:

Bounding box
Confidence score
Person class

Example:

person 0.95

This can be used in:

Security systems
Crowd monitoring
Smart surveillance
People counting
Workplace monitoring

6. 🕺 Pose Detection

Pose Detection identifies important human body keypoints and represents the human body as a skeleton.

The model can detect body landmarks such as:

Head
Shoulders
Elbows
Wrists
Hips
Knees
Ankles

Example:

       Head
         ●
       /   \
      ●     ●
     /       \
    ●         ●
     \       /
      ●     ●

Pose Detection can be useful for:

Fitness applications
Sports analysis
Human activity recognition
Gesture analysis
Exercise monitoring
Human-computer interaction

7. 🌈 Semantic Segmentation

Semantic Segmentation assigns a class to different regions of an image.

Instead of simply detecting an object with a bounding box, the system classifies pixels according to their semantic category.

For example:

Road
Car
Person
Building
Sky
Vegetation

This provides a detailed understanding of the scene.

🧠 YOLO Models Explored

During the project, I experimented with multiple YOLO model versions.

YOLOv8

Used to understand fundamental YOLO computer vision workflows.

YOLO11

Used for exploring newer YOLO capabilities and computer vision tasks.

YOLO26

Used in the final pose-detection application.

The experiments helped me understand that different YOLO models and task-specific model variants can be used for different computer vision requirements.

🛠️ Technologies Used
Programming Language
Python
Computer Vision
OpenCV
YOLO
Ultralytics
AI / Machine Learning
YOLOv8
YOLO11
YOLO26
Web Application
Gradio
Video Processing
OpenCV
FFmpeg-compatible video workflows
yt-dlp for supported video URL downloading

🌐 Final Web Application

After completing the individual YOLO experiments, I converted the YOLO Pose Detection implementation into an interactive web application using Gradio.

The application provides multiple ways to provide video input.

📤 1. Upload Video

Users can upload a video directly from their computer.

Computer
   ↓
Upload Video
   ↓
Gradio Application
   ↓
YOLO Pose Detection
   ↓
Processed Video
📷 2. Webcam Recording

Users can record a video directly through their browser webcam.

Webcam
   ↓
Record Video
   ↓
YOLO Pose Detection
   ↓
Annotated Video

This makes the application interactive instead of requiring users to manually prepare a video file.

🔗 3. Video URL

The application also provides a video URL input.

A supported URL can be provided to the application.

Video URL
    ↓
Video Download
    ↓
YOLO Processing
    ↓
Pose Detection
    ↓
Annotated Video

The application uses:

requests for suitable direct video URLs
yt-dlp for supported video platforms

Note: Not every website URL is guaranteed to work. Private, restricted, authentication-protected, or unsupported videos may not be downloadable.

⚡ YOLO Pose Detection Workflow

The final application follows this workflow:

          VIDEO INPUT
               │
      ┌────────┼────────┐
      │        │        │
   Upload   Webcam     URL
      │        │        │
      └────────┼────────┘
               ↓
        Video Processing
               ↓
        OpenCV Frame Read
               ↓
         YOLO Pose Model
               ↓
        Pose Detection
               ↓
       Annotated Frames
               ↓
        Output MP4 Video
               ↓
       View / Download
🎥 Frame-by-Frame Processing

The video is processed frame by frame.

The basic workflow is:

ret, frame = cap.read()

results = model(frame)

annotated_frame = results[0].plot()

writer.write(annotated_frame)

Each frame is passed through the YOLO model.

The detection results are then drawn onto the frame and written into the output video.

📊 Application Features
Feature	Supported
Image/Video Computer Vision	✅
Object Detection	✅
Object Counting	✅
Object Tracking	✅
Object Segmentation	✅
Person Detection	✅
Pose Detection	✅
Semantic Segmentation	✅
YOLOv8	✅
YOLO11	✅
YOLO26	✅
Video Upload	✅
Webcam Recording	✅
Video URL Input	✅
Annotated Video Output	✅
Download Processed Video	✅
Gradio Interface	✅

The exact filenames can be changed according to the files used in your local project.

⚙️ Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

Navigate into the project:

cd YOLO

Install the required packages:

pip install -r requirements.txt
📦 Requirements

The main dependencies are:

gradio
ultralytics
opencv-python-headless
requests
yt-dlp
▶️ Run the Application Locally

Run:

python app.py

The Gradio application will provide a local URL similar to:

http://127.0.0.1:7860

Open the URL in your browser.

☁️ Deployment

The application can be deployed using platforms that support Python and Gradio applications.

The project can be hosted on:

Hugging Face Spaces
Other Python-compatible cloud platforms

For Hugging Face Spaces, the main files are:

app.py
requirements.txt
yolo26n-pose.pt

The Hugging Face Space installs the required dependencies and runs the Gradio application.

🔐 Model File

The YOLO model weights are required for inference.

For the final pose-detection application:

yolo26n-pose.pt

should be available to the application.

The model path is configured in Python:

MODEL_PATH = "yolo26n-pose.pt"

model = YOLO(MODEL_PATH)
🚀 Real-World Applications

The concepts explored in this project can be applied to many real-world systems.

Object Detection
Traffic monitoring
Retail systems
Industrial inspection
Smart surveillance
Object Tracking
Vehicle tracking
People tracking
Sports analytics
Traffic analysis
Pose Detection
Fitness applications
Sports analysis
Exercise monitoring
Gesture recognition
Segmentation
Autonomous systems
Medical imaging
Industrial inspection
Scene understanding
📚 What I Learned

Through this project, I gained practical experience with:

YOLO-based Computer Vision
Object Detection
Object Counting
Object Tracking
Object Segmentation
Person Detection
Pose Estimation
Semantic Segmentation
OpenCV video processing
Frame-by-frame inference
Video input/output handling
Webcam integration
URL-based video processing
Gradio application development
AI model integration into applications

Most importantly, I learned how to move from individual computer-vision experiments to a practical user-facing AI application.

🙏 Mentor

Special thanks to my mentor:

Kodi Prakash Senapati Sir

for his guidance, support, and encouragement throughout this learning journey.

🔮 Future Improvements

Possible future enhancements include:

Real-time webcam pose detection
Live video streaming
More YOLO models
Object-specific analytics
Pose-based activity recognition
Real-time FPS display
Detection statistics
Confidence threshold controls
GPU acceleration
Cloud deployment optimization
Improved video processing speed
More input sources
👨‍💻 Author

Pavan Kumar Torlapati

Python Developer | AI & Generative AI Enthusiast | Computer Vision

Skills Demonstrated
Python
YOLO
Computer Vision
OpenCV
Machine Learning
AI Applications
Video Processing
Gradio
⭐ Conclusion

This project demonstrates my practical exploration of modern YOLO-based Computer Vision, starting from individual tasks such as detection, counting, tracking, segmentation, and pose estimation and progressing toward an interactive AI application.

The final application demonstrates how a trained computer-vision model can be integrated into a user-friendly interface where users can provide video through upload, webcam recording, or supported URLs, process it using YOLO Pose Detection, and download the resulting annotated video.

If you find this project useful or interesting, feel free to explore the repository and connect with me.

⭐ Thank you for visiting the project!
