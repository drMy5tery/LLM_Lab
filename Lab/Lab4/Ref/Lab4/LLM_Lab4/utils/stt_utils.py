import speech_recognition as sr

def transcribe_audio(audio_path, lang_choice):
    """
    Transcribe the audio file to text using Google Speech Recognition API.

    Args:
        audio_path (str): Path to the audio WAV file.
        lang_choice (str): Language name - "English", "Tamil", or "French".

    Returns:
        str: Transcribed text or error message.
    """
    recognizer = sr.Recognizer()
    language_map = {
        "English": "en-US",
        "Tamil": "ta-IN",
        "French": "fr-FR"
    }

    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language=language_map.get(lang_choice, "en-US"))
        return text
    except sr.RequestError as e:
        return f"Error: API unavailable or unresponsive - {e}"
    except sr.UnknownValueError:
        return "Error: Unable to recognize speech"
    except Exception as e:
        return f"Error transcribing: {str(e)}"
