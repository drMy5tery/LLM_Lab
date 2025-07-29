import speech_recognition as sr

def transcribe_audio(audio_path, lang_choice):
    recognizer = sr.Recognizer()
    lang_map = {"English": "en-US", "Tamil": "ta-IN", "French": "fr-FR"}
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language=lang_map[lang_choice])
    except Exception as e:
        return f"Error transcribing: {str(e)}"
