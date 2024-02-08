from transformers import pipeline
from gtts import gTTS
import os

import sounddevice as sd
import speech_recognition as sr

import whisper

import sounddevice as sd
import speech_recognition as sr
import numpy as np



def voice_to_text():
    # Set up parameters for recording audio
    sample_rate = 44100
    duration = 5  # Number of seconds to record

    print("Listening...")
    # Record audio
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    print(audio.shape)

    # Convert audio to text
    recognizer = sr.Recognizer()
    audio_data = np.squeeze(audio)  # Convert to 1D array

    print(audio_data.shape)
    try:
        text = recognizer.recognize_google(audio_data, language='en-US')
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        return None


def chat_with_model(input_text):
    # Load conversational model from Hugging Face
    chat_pipeline = pipeline("conversation", model="mistralai/Mistral-7B-v0.1")

    # Chat with the model
    response = chat_pipeline(input_text)
    return response


def text_to_voice(text):
    # Load text-to-speech model from Hugging Face
    tts_pipeline = pipeline("text2speech", model="openai/whisper-large-v3")

    # Convert text to speech
    audio_data = tts_pipeline(text)[0]["audio"]

    with open("response.wav", "wb") as file:
        file.write(audio_data)

    os.system("afplay response.wav")  # Use 'afplay' to play audio on Mac


def main():
    model1 = whisper.load_model("base")
    model1.transcribe("audio.mp3")

    return

    while True:
        # Step 1: Voice to Text
        input_text = voice_to_text()
        if input_text is None:
            continue

        # Step 2: Chat with Model
        response = chat_with_model(input_text)

        # Step 3: Text to Voice
        text_to_voice(response[0]['generated_text'])


if __name__ == "__main__":
    main()
