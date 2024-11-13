# Import necessary libraries
import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import base64
import json
import requests

from PIL import Image
import tempfile
import os
import whisper
import subprocess
import re
from io import BytesIO
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from dotenv import load_dotenv

# Load environment variables (if using .env)
load_dotenv()
def generate_image_from_text(prompt):
    url = "https://api.wizmodel.com/sdapi/v1/txt2img"
    api_key_stable_diffusion = os.getenv('STABLE_DIFF_API')  # Replace with your API key if not using .env
    
    payload = json.dumps({
        "prompt": prompt,
        "steps": 50
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key_stable_diffusion}'
    }

    # Make the request to the API
    response = requests.post(url, headers=headers, data=payload)
    print(response.json())
    # Check for a successful response
    if response.status_code == 200:
        try:
            # Decode base64 image from response
            base64_string = response.json()['images'][0]
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
            return image
        except (KeyError, ValueError):
            st.error("Error in API response format.")
            return None
    else:
        st.error("Failed to retrieve image. Check API key or network connection.")
        return None
# Load a pre-trained Haar Cascade for detecting faces (used as a proxy for people detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Helper Functions
def extract_keywords(title, details):
    text = title + " " + details
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    return list(set(keywords))

def convert_video_to_audio(video_path, audio_path="audio.wav"):
    subprocess.run(['ffmpeg', '-i', video_path, '-ac', '1', '-ar', '16000', audio_path, '-y'])
    return audio_path

def transcribe_audio_with_whisper(audio_path, model_name="base"):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result

def extract_key_moments(transcription, keywords):
    key_moments = []
    keywords_lower = [kw.lower() for kw in keywords]
    for segment in transcription["segments"]:
        text = segment["text"].lower()
        if any(keyword in text for keyword in keywords_lower):
            key_moments.append(segment["start"])
    return key_moments

def extract_keyframes_scenedetect(video_path):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30))
    scene_manager.detect_scenes(video)
    return [scene[0] for scene in scene_manager.get_scene_list()]

def get_frame_at_time(video_path, timestamp):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def calculate_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces)

def rank_keyframes(keyframes, video_path):
    ranked_keyframes = []
    for timestamp in keyframes:
        frame = get_frame_at_time(video_path, timestamp)
        if frame is not None:
            sharpness_score = calculate_sharpness(frame)
            people_count = detect_faces(frame)
            score = sharpness_score + (people_count * 10)
            ranked_keyframes.append((timestamp, score, frame, sharpness_score, people_count))
    ranked_keyframes.sort(key=lambda x: x[1], reverse=True)
    return ranked_keyframes[:5]

# Streamlit app
st.title("Video Keyframe Detection Tool")

# File uploader for video
uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
keyframes = []
# Option to choose analysis type
analysis_type = st.selectbox("Choose Analysis Type", ["Emotion Detection-Based", "Subtitle Keywords-Based"])

if uploaded_video is not None:
    # Save video to temporary file
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    if analysis_type == "Emotion Detection-Based":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        frame_interval = 30
        frame_count = 0
        highest_emotion_frames = {emotion: {"score": 0, "timestamp": 0} for emotion in [
            "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(30, 30))
                if len(faces) > 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    try:
                        result = DeepFace.analyze(img_path=rgb_frame, actions=['emotion'], enforce_detection=False)
                        for emotion, score in result[0]["emotion"].items():
                            if score > highest_emotion_frames[emotion]["score"]:
                                highest_emotion_frames[emotion] = {"score": score, "timestamp": frame_count / cap.get(cv2.CAP_PROP_FPS)}
                    except Exception as e:
                        st.write("Error analyzing frame:", e)
            frame_count += 1
        cap.release()

        keyframes = [data["timestamp"] for data in highest_emotion_frames.values() if data["score"] > 0]
        st.write("Analyzing video for emotion-based keyframes...")

    elif analysis_type == "Subtitle Keywords-Based":
        title = st.text_input("Enter Video Title")
        details = st.text_area("Enter Video Details")
        if title and details:
            keywords = extract_keywords(title, details)
            audio_path = convert_video_to_audio(video_path)
            transcription = transcribe_audio_with_whisper(audio_path)
            keyframes = extract_key_moments(transcription, keywords)
            st.write("Extracted Keywords:", keywords)

    # Ranking and displaying keyframes
    if keyframes:
        ranked_keyframes = rank_keyframes(keyframes, video_path)
        st.write("Top 5 Ranked Keyframes:")
        for i, (timestamp, score, frame_image, sharpness_score, people_count) in enumerate(ranked_keyframes):
            st.image(frame_image, caption=f"Keyframe {i+1} - Time: {timestamp}s, Score: {score:.2f}, Sharpness: {sharpness_score:.2f}, Faces Detected: {people_count}")

            # Add download button for each keyframe
            keyframe_img = Image.fromarray(frame_image)
            img_bytes = BytesIO()
            keyframe_img.save(img_bytes, format="JPEG")
            img_bytes.seek(0)
            st.download_button(
                label=f"Download Keyframe {i+1}",
                data=img_bytes,
                file_name=f"keyframe_{i+1}.jpg",
                mime="image/jpeg"
            )

    os.remove(video_path)
st.title("Text-to-Image Generation Tool")

# Prompt input
prompt = st.text_input("Enter your text prompt", "Christiano Ronaldo, the football player winning a world cup")

# Button to generate the image
if st.button("Generate Image"):

    st.write("Generating image...")
    image = generate_image_from_text(prompt)  # Call the defined function
    
    if image:
        # Display generated image
        st.image(image, caption="Generated Image", use_column_width=True)
        
        # Create a download button for the generated image
        img_bytes = BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        st.download_button(
            label="Download Generated Image",
            data=img_bytes,
            file_name="generated_image.jpg",
            mime="image/jpeg"
        )
    else:
        st.error("Image generation failed. Please check your prompt and API key.")

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from rembg import remove

# Function to enhance image using OpenCV and Pillow
def enhance_image(image, denoise=True, contrast=True, brightness=True, sharpness=True, upscale_factor=2):
    # Convert the PIL image to an RGB NumPy array for processing
    open_cv_image = np.array(image)
    
    if denoise:
        open_cv_image = cv2.GaussianBlur(open_cv_image, (5, 5), 0)

    if contrast:
        ycrcb_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2YCrCb)
        channels = list(cv2.split(ycrcb_image))
        channels[0] = cv2.equalizeHist(channels[0])
        ycrcb_image = cv2.merge(channels)
        open_cv_image = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2RGB)  # Convert back to RGB

    if brightness:
        open_cv_image = cv2.convertScaleAbs(open_cv_image, alpha=1.2, beta=50)

    if sharpness:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        open_cv_image = cv2.filter2D(open_cv_image, -1, kernel)

    if upscale_factor > 1:
        height, width = open_cv_image.shape[:2]
        new_dimensions = (int(width * upscale_factor), int(height * upscale_factor))
        open_cv_image = cv2.resize(open_cv_image, new_dimensions, interpolation=cv2.INTER_CUBIC)

    enhanced_image = Image.fromarray(open_cv_image, 'RGB')
    return enhanced_image

# Function to cartoonize image
def cartoonize_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))

# Function to insert stickers and remove background
def insert_sticker(image, background_image):
    image = remove(image)
    st.image(image, caption="Removed Background", use_column_width=True)
    
    width = st.slider("Select the width of the sticker", 50, background_image.width, image.width)
    height = st.slider("Select the height of the sticker", 50, background_image.height, image.height)
    resized_image = image.resize((width, height))
    
    position = st.selectbox("Select the position of the sticker on the background", ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"])
    bg_width, bg_height = background_image.size
    img_width, img_height = resized_image.size
    
    if position == "Top-Left":
        pos = (0, 0)
    elif position == "Top-Right":
        pos = (bg_width - img_width, 0)
    elif position == "Bottom-Left":
        pos = (0, bg_height - img_height)
    elif position == "Bottom-Right":
        pos = (bg_width - img_width, bg_height - img_height)
    else:  # Center
        pos = ((bg_width - img_width) // 2, (bg_height - img_height) // 2)
    
    background_image.paste(resized_image, pos, resized_image)
    st.image(background_image, caption="Final Image with Sticker", use_column_width=True)

    # Download button for stickerified image
    sticker_image_bytes = BytesIO()
    background_image.save(sticker_image_bytes, format="JPEG")
    sticker_image_bytes.seek(0)
    st.download_button(
        label="Download Stickerified Image",
        data=sticker_image_bytes,
        file_name="stickerified_image.jpg",
        mime="image/jpeg"
    )

# Function to add text overlay using OpenCV
def add_text_overlay_opencv(image, text, position=(50, 50), font_size=1, color=(255, 255, 255), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    image_np = np.array(image)
    cv2.putText(image_np, text, position, font, font_size, color, thickness, cv2.LINE_AA)
    return Image.fromarray(image_np)

# Function to add logo overlay
def add_logo_overlay(image, logo_url, position="bottom-right", logo_size=(100, 100)):
    if not logo_url:
        st.error("Please enter a valid logo URL.")
        return image
    
    response = requests.get(logo_url)

    if response.status_code != 200:
        st.error("Failed to fetch logo image. Please check the URL.")
        return image
    
    logo = Image.open(BytesIO(response.content))
    logo = logo.convert('RGB')
    logo_width = st.slider("Logo Width", 20, 10000, logo_size[0])
    logo_height = st.slider("Logo Height", 20, 10000, logo_size[1])
    logo = logo.resize((logo_width, logo_height))

    if position == "top-left":
        logo_position = (10, 10)
    elif position == "top-right":
        logo_position = (image.width - logo.width - 10, 10)
    elif position == "bottom-left":
        logo_position = (10, image.height - logo.height - 10)
    else:
        logo_position = (image.width - logo.width - 10, image.height - logo.height - 10)

    image.paste(logo, logo_position, logo if logo.mode == "RGBA" else None)
    return image

# Streamlit app
def main():
    st.title("Image Enhancement & Manipulation Tool")

    # Upload the image
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

    # User option for custom image upload
    if not image:
        st.warning("Please upload an image to proceed.")
        return

    # Enhancement options
    st.sidebar.header("Enhancement Options")
    denoise = st.sidebar.checkbox("Apply Denoise (Gaussian Blur)", value=True)
    contrast = st.sidebar.checkbox("Improve Contrast (Histogram Equalization)", value=True)
    brightness = st.sidebar.checkbox("Adjust Brightness", value=True)
    sharpness = st.sidebar.checkbox("Apply Sharpening Filter", value=True)
    upscale_factor = st.sidebar.slider("Upscale Factor", min_value=1, max_value=5, value=2, step=1)

    # Enhance image
    enhanced_image = enhance_image(image, denoise, contrast, brightness, sharpness, upscale_factor)
    print(enhanced_image.height)
    st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)

    # Provide download option for enhanced image
    enhanced_image_bytes = BytesIO()
    enhanced_image.save(enhanced_image_bytes, format="JPEG")
    enhanced_image_bytes.seek(0)
    st.sidebar.download_button(
        label="Download Enhanced Image",
        data=enhanced_image_bytes,
        file_name="enhanced_image.jpg",
        mime="image/jpeg"
    )

    # Select further manipulation options
    st.sidebar.header("Choose Next Action")
    action = st.sidebar.radio("What would you like to do next?", ["Cartoonify", "Insert Sticker", "Add Text Overlay", "Add Logo Overlay"])

    
    if action == "Cartoonify":
        cartoon_image = cartoonize_image(enhanced_image)
        st.image(cartoon_image, caption="Cartoonized Image", use_column_width=True)

        # Download button for cartoonified image
        cartoon_image_bytes = BytesIO()
        cartoon_image.save(cartoon_image_bytes, format="JPEG")
        cartoon_image_bytes.seek(0)
        st.download_button(
            label="Download Cartoonified Image",
            data=cartoon_image_bytes,
            file_name="cartoonified_image.jpg",
            mime="image/jpeg"
        )

    elif action == "Insert Sticker":
        background_file = st.file_uploader("Upload a background image", type=["png", "jpg", "jpeg"])
        
    
        if background_file is not None:
            background_image = Image.open(background_file)
            insert_sticker(enhanced_image, background_image)




    elif action == "Add Text Overlay":
        text_overlay = st.text_input("Enter text to overlay on the image", "Your Text Here")
        font_size = st.slider("Font size of overlay text", min_value=1, max_value=int(enhanced_image.height/50), value=int(enhanced_image.height/100))
        thickness = st.slider("Text Thickness", min_value=1, max_value=300, value=2)
        color = st.color_picker("Pick text color", "#FFFFFF")
        font_type = st.selectbox("Select font type", [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX])
        position = st.selectbox("Select the position of the sticker on the background", ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"])
        bg_width, bg_height = enhanced_image.size
        image_np = np.array(image)
        cv2.putText(image_np, text_overlay, (50, 50), font_type, font_size, (255, 255, 255), thickness, cv2.LINE_AA)
        overlay_image = Image.fromarray(image_np)

        img_width, img_height = overlay_image.size
        
        if position == "Top-Left":
            pos = (0, 0+img_height//2)
        elif position == "Top-Right":
            pos = (bg_width-img_width, 0+img_height//2)
        elif position == "Bottom-Left":
            pos = (0, bg_height)
        elif position == "Bottom-Right":
            pos = (bg_width-img_width, bg_height)
        else:  # Center
            pos = ((bg_width) // 2 - img_width// 2, (bg_height) // 2)
        overlay_image = add_text_overlay_opencv(enhanced_image,  text_overlay,position=pos, font_size=font_size, thickness=thickness, color=tuple(int(color[i:i+2], 16) for i in (1, 3, 5)), font=font_type)
        st.image(overlay_image, caption="Image with Text Overlay", use_column_width=True)

        # Download button for image with text overlay
        overlay_image_bytes = BytesIO()
        overlay_image.save(overlay_image_bytes, format="JPEG")
        overlay_image_bytes.seek(0)
        st.download_button(
            label="Download Image with Text Overlay",
            data=overlay_image_bytes,
            file_name="text_overlay_image.jpg",
            mime="image/jpeg"
        )

    elif action == "Add Logo Overlay":
        logo_url = st.text_input("Enter the logo URL")
        position = st.selectbox("Select logo position", ["top-left", "top-right", "bottom-left", "bottom-right"])
        logo_image = add_logo_overlay(enhanced_image, logo_url, position)
        st.image(logo_image, caption="Image with Logo Overlay", use_column_width=True)

        # Download button for image with logo overlay
        logo_image_bytes = BytesIO()
        logo_image.save(logo_image_bytes, format="JPEG")
        logo_image_bytes.seek(0)
        st.download_button(
            label="Download Image with Logo Overlay",
            data=logo_image_bytes,
            file_name="logo_overlay_image.jpg",
            mime="image/jpeg"
        )
    

if __name__ == "__main__":
    main()