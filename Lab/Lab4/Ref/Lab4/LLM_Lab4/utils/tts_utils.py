from gtts import gTTS
import tempfile
import os
import pygame
import time

def speak_text(text, lang="en"):
    if not text or text.strip() == "":
        return
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        tmp_path = tmp.name

    pygame.mixer.init()
    pygame.mixer.music.load(tmp_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.5)

    pygame.mixer.quit()
    os.remove(tmp_path)
