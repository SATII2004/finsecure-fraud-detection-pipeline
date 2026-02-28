import pyttsx3
import threading

def trigger_voice_alert(message):
    """
    Runs the voice alert in a separate thread.
    Includes error handling for Cloud environments (Linux) 
    where audio drivers are missing.
    """
    def speak():
        try:
            # Attempt to initialize the engine
            engine = pyttsx3.init()
            engine.setProperty('rate', 150) 
            engine.say(message)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            # On Streamlit Cloud, this will log the error to the console 
            # instead of crashing the web app.
            print(f"Voice Alert Log: Audio output not supported on this system. {e}")

    # Threading prevents the UI from freezing while the 'voice' is speaking
    alert_thread = threading.Thread(target=speak)
    alert_thread.daemon = True # Ensures thread closes when app stops
    alert_thread.start()