import pyttsx3
import threading

def trigger_voice_alert(message):
    """
    Runs the voice alert in a separate thread to prevent 
    the Dashboard from freezing.
    """
    def speak():
        engine = pyttsx3.init()
        # Setting a slightly slower rate for a 'serious' warning tone
        engine.setProperty('rate', 150) 
        engine.say(message)
        engine.runAndWait()
        engine.stop()

    # Threading is necessary because TTS engines are blocking
    alert_thread = threading.Thread(target=speak)
    alert_thread.start()