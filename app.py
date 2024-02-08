import torch
import speech_recognition as sr
from transformers import pipeline, Conversation
import winsound
from gtts import gTTS
import os


def check_gpu():
    if torch.cuda.is_available():
        print("GPU is available. Moving models to GPU...")
        torch_device = "cuda"
    else:
        print("GPU is not available. Using CPU...")
        torch_device = "cpu"
    return torch_device


def voice_to_text(recognizer):
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            return None


def text_to_voice(text, torch_device):
    # Load text-to-speech model from Hugging Face
    tts_pipeline = pipeline("text-to-speech", model="openai/whisper-large-v3", device=torch_device)

    # Convert text to speech
    audio_data = tts_pipeline(text)[0]["audio"]

    with open("response.wav", "wb") as file:
        file.write(audio_data)

        # Play the audio
        winsound.PlaySound("response.wav", winsound.SND_FILENAME)


def main():
    torch_device = check_gpu()

    # Load conversational model from Hugging Face
    print("Loading Mistral")
    chatbot = pipeline("conversational", model="mistralai/Mistral-7B-v0.1", device="cpu",
                             batch_size=1)
    recognizer = sr.Recognizer()

    previous_response = ""  # Initialize previous response

    i = 0
    while True:
        # Step 1: Voice to Text
        input_text = voice_to_text(recognizer)
        if input_text is None:
            continue

        if i == 0:
            conversation = Conversation(input_text)
            conversation = chatbot(conversation)
            conversation.generated_responses[-1]
            i = 1
        else:
            conversation.add_user_input(input_text)
            conversation = chatbot(conversation)
            conversation.generated_responses[-1]

        # print the conversation
        print(conversation)

        # Step 3: Text to Voice
        # text_to_voice(response, torch_device)


if __name__ == "__main__":
    main()
