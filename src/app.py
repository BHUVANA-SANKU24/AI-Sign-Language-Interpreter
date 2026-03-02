import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import cv2
import threading
import speech_recognition as sr
import pyttsx3
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "static_gesture_cnn.h5")
LABEL_PATH = os.path.join(BASE_DIR, "models", "labels.npy")

from src.generation.sign_animator import SignAnimator
from src.recognition.gesture_recognition import GestureRecognizer

# ---------------- INIT ----------------
sign_animator = SignAnimator()
sign_recognizer = GestureRecognizer(
    model_path=MODEL_PATH,
    label_encoder_path=LABEL_PATH,
    confidence_threshold=0.7
)

engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

st.set_page_config(page_title="AI Sign Language Interpreter", layout="wide")
st.title("🤟 AI Sign Language Interpreter")

mode = st.radio(
    "Select Mode",
    ["Sign → Text", "Text → Sign + Voice", "Audio → Sign"]
)

# ---------------- SIGN → TEXT ----------------
if mode == "Sign → Text":
    st.info("Click start to open camera")

    if "run_camera" not in st.session_state:
        st.session_state.run_camera = False

    if st.button("Start Camera"):
        st.session_state.run_camera = True

    if st.button("Stop Camera"):
        st.session_state.run_camera = False

    frame_area = st.image([])

    if st.session_state.run_camera:
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        if ret:
            result = sign_recognizer.predict(frame)
            st.write(f"Detected: **{result}**")
            frame_area.image(frame, channels="BGR")

        cap.release()

# ---------------- TEXT → SIGN ----------------
elif mode == "Text → Sign + Voice":
    text = st.text_input("Enter Text")

    if st.button("Generate Sign"):
        if text.strip():
            st.success(f"Generating sign for: {text}")
            sign_animator.generate_from_text(text)
            threading.Thread(target=speak, args=(text,)).start()
        else:
            st.warning("Please enter text")

# ---------------- AUDIO → SIGN ----------------
elif mode == "Audio → Sign":
    if st.button("Record Audio"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            st.success(f"Recognized: {text}")
            sign_animator.generate_from_text(text)
            threading.Thread(target=speak, args=(text,)).start()
        except:
            st.error("Could not recognize audio")