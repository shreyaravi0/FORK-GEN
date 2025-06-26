# 🎬✨ Video Thumbnail Generator 🎨📸

Welcome to the **Video Thumbnail Generator**! 🌟 This tool leverages **Machine Learning** 🧠, **Scene Detection** 📹, and **Emotion Analysis** 😃😢 to help you find the most impactful keyframes for thumbnails. It also includes **Text-to-Image Generation** 🌌🖼️ to create unique visuals from prompts! Whether you’re a content creator or an editor, this tool is designed to make your work smoother and more creative!.🚀

---

## 🌈 Features

- **🎥 Keyframe Detection:** Detects important frames based on **emotion analysis** 😮 or **subtitle keywords** 📝.
- **🎨 Text-to-Image Generation:** Generate stunning visuals from text prompts using **Stable Diffusion** ✍️.
- **🔍 Advanced Image Processing:** Enhances, cartoonizes, and customizes images with **OpenCV** 💻 and **Pillow** 🖌️.
- **🎭 Face Detection and Emotion Analysis:** Identify the **strongest emotions** in frames for attention-grabbing thumbnails! 🔥

---

## 🛠️ Tech Stack

| Tech             | Purpose                                    |
|------------------|--------------------------------------------|
| `Streamlit` 🌐   | Interactive user interface for easy use    |
| `OpenCV` 📷      | Image processing and frame extraction      |
| `DeepFace` 😊    | Emotion detection for impactful keyframes  |
| `Stable Diffusion` 🎨 | Text-to-image generation with API integration |
| `Whisper` 🎙️     | Audio transcription and keyword extraction |
| `SceneDetect` 🔍 | Scene analysis for key moments             |
| `NLTK` 🧩         | Keyword extraction from subtitles         |
| `rembg` ✂️       | Background removal for custom stickers     |

---

## 📝 Requirements

Here’s what you need to get started:

### 📥 Install Dependencies

Ensure you have the following packages installed. Run this in your terminal:
```bash
pip install streamlit opencv-python-headless deepface pillow requests whisper scenedetect nltk rembg python-dotenv
```

> **Note**: Some libraries may require additional installations. See individual docs for setup instructions! 🛠️

### 🔑 Environment Variables

Set up your `.env` file for API keys! 🔐 

```plaintext
STABLE_DIFF_API=your_stable_diffusion_api_key_here
```

### 🧠 Models and Data Files

Some libraries like `Whisper` and `DeepFace` may download their model files on the first run. 📥 

---
make sure to make a .env file to put in the api key.

## 📸 Usage

### 🌟 Upload a Video 🎞️
1. Upload your video (mp4, mov, avi formats).
2. Choose an **Analysis Type**:
   - **Emotion Detection-Based** 😲😃
   - **Subtitle Keywords-Based** 📝

### 🔍 Keyframe Extraction
- **Emotion Detection-Based**: Detects emotions in frames.
- **Subtitle Keywords-Based**: Searches audio transcription for keywords and ranks frames accordingly.

### 📐 Customize and Download
- **Ranked Keyframes**: Preview and download top-ranked frames.
- **Text-to-Image**: Enter prompts to generate custom images from text! ✍️

---

## 💡 Key Functionalities

### `generate_image_from_text(prompt)`
- Generates an image based on your text prompt using **Stable Diffusion API**! 🧑‍🎨

### `extract_keywords(title, details)`
- Uses **NLTK** to extract keywords from titles and descriptions.

- 

### `convert_video_to_audio(video_path)`
- Converts video to audio using **FFmpeg** for transcription! 🎙️

### `transcribe_audio_with_whisper(audio_path)`
- Transcribes audio using the **Whisper** model for accurate speech-to-text! ✏️

### **project teammates**
This project was collaborated with Poorvi Bellur, Ansh Kashyap and Aditya K. I am extremely gratfeul for their support without which we would not have won 2nd place in the Craft and Code hackathon conducted by IIIT bhubaneshwar.

