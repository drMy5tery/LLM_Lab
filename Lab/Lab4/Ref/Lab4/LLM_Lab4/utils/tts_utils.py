from gtts import gTTS
from playsound import playsound
import tempfile
import os

def speak_text(text, lang="en"):
    if not text or text.strip() == "":
        return
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        playsound(tmp.name)
        os.remove(tmp.name)

