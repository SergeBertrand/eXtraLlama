import gradio as gr
import pyttsx3
import speech_recognition as sr
import subprocess
import threading
import requests
import json
import os
from typing import List, Tuple
import ctypes

# Utilisation de constantes pour éviter les recomptages inutiles
AVAILABLE_MODELS = [
    "llama3.1", "gemma2", "mistral-nemo", "starling-lm"
]

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def open_task_manager():
    if is_admin():
        # Exécution du gestionnaire des tâches directement si déjà en mode admin
        return (subprocess.Popen('taskmgr.exe'), "Le gestionnaire des tâches a été ouvert.")
    else:
        # Demande d'élévation des droits si pas en mode admin
        ctypes.windll.shell32.ShellExecuteW(None, "runas", "taskmgr.exe", None, None, 1)
        return (None, "Demande d'élévation des droits envoyée pour ouvrir le gestionnaire des tâches.")

def open_folder(folder_path):
    resolved_path = os.path.expandvars(folder_path)
    try:
        subprocess.Popen(['explorer', resolved_path])
        return "Dossier ouvert."
    except Exception as e:
        return f"Erreur lors de l'ouverture du dossier : {e}"

COMMANDS = {
# commande test
    'bonjour': lambda: (None, "Comment puis-je vous aider aujourd'hui?"),

# commandes des applications    
    'ouvre la calculette': lambda: (subprocess.Popen('calc.exe'), "La calculette a été ouverte."),
    'ouvre le bloc-notes': lambda: (subprocess.Popen('notepad.exe'), "Le bloc-notes a été ouvert."),
    'ouvre outlook': lambda: (subprocess.Popen(r'C:\Program Files\Microsoft Office\root\Office16\OUTLOOK.EXE'), "outlook a été ouvert."),
    'mets de la musique': lambda: (play_music(), "La playlist a été ouverte dans VLC."),

# commandes Windows
    'ouvre le gestionnaire des tâches': open_task_manager,    
    'ouvre le panneau des configurations': lambda: (subprocess.Popen('control.exe')),
    'verrouille la session': lambda: (subprocess.Popen('rundll32.exe user32.dll, LockWorkStation')),
    'ouvre les téléchargements': lambda: (None, open_folder(r"%USERPROFILE%\Downloads")),
    'ouvre le dossier des documents': lambda: (None, open_folder(r"%USERPROFILE%\Documents")),
    'ouvre le dossier des images': lambda: (None, open_folder(r"%USERPROFILE%\Pictures")),
    'ouvre le dossier vidéo': lambda: (None, open_folder(r"%USERPROFILE%\Videos")),
    'ouvre le dossier utilisateur': lambda: (None, open_folder(r"%USERPROFILE%")),
# commandes Web
    'ouvre youtube': lambda: (subprocess.Popen('start iexplore.exe https://www.youtube.com/feed/subscriptions/', shell=True), "youtube a été ouvert dans le navigateur."),
    'ouvre facebook': lambda: (subprocess.Popen('start iexplore.exe https://www.facebook.com/?sk=h_chr', shell=True), "facebook a été ouvert dans le navigateur."),
    'ouvre plex': lambda: (subprocess.Popen('start iexplore.exe https://app.plex.tv/desktop/#!/', shell=True), "plex a été ouvert dans le navigateur."),

# commandes télé
    'ouvre france 24': lambda: (subprocess.Popen('start iexplore.exe https://www.youtube.com/embed/l8PMl7tUDIE?autoplay=1&mute=1', shell=True), "france 24 a été ouverte dans le navigateur."),
    'ouvre cnews': lambda: (subprocess.Popen('start iexplore.exe https://www.dailymotion.com/embed/video/x3b68jn?autoplay=1', shell=True), "cnews a été ouverte dans le navigateur."),
}

def play_music():
    playlist_path = os.path.expandvars(r'%userprofile%\Music\playlist.m3u')
    vlc_path = r'C:\Program Files\VideoLAN\VLC\vlc.exe'
    if os.path.isfile(playlist_path) and os.path.isfile(vlc_path):
        subprocess.Popen([vlc_path, playlist_path])
    else:
        raise FileNotFoundError("Le fichier de playlist ou VLC n'a pas été trouvé.")

def remove_repeated_phrases(text: str) -> str:
    seen = set()
    result = []
    for word in text.split():
        if word not in seen:
            seen.add(word)
            result.append(word)
    return ' '.join(result)

def generate_response(msg: str, history: List[Tuple[str, str]], model: str,
                      temperature=0.8, num_predict=1024, top_k=40, top_p=0.92,
                      presence_penalty=0.5, repeat_penalty=1.5, frequency_penalty=1.8, num_ctx=7168, num_keep=3072):
    chat_history = [{"role": "user", "content": msg}]
    for query, response in history:
        chat_history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ])
    if not is_model_installed(model):
        handle_model_selection(model)
        if not is_model_installed(model):
            AVAILABLE_MODELS.remove(model)
            return f"Erreur : Le modèle {model} n'a pas pu être téléchargé."

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": chat_history,
                "options": {
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "presence_penalty": presence_penalty,
                    "repeat_penalty": repeat_penalty,
                    "frequency_penalty": frequency_penalty,
                    "num_predict": num_predict,
                    "num_ctx": num_ctx,
                    "num_keep": num_keep
                }
            },
            timeout=60,
            stream=True
        )
        response.raise_for_status()

        full_content = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    part = json.loads(line)
                    if 'message' in part and 'content' in part['message']:
                        full_content += part['message']['content']
                except json.JSONDecodeError:
                    return f"Erreur lors de la réponse du modèle : {line}"

        return remove_repeated_phrases(full_content.strip())

    except requests.RequestException as e:
        return f"Erreur lors de la génération de la réponse : {str(e)}"


def get_installed_models():
    try:
        models = subprocess.check_output(["ollama", "list"], stderr=subprocess.DEVNULL, universal_newlines=True)
        return [model.split(":")[0].strip() for model in models.splitlines() if not model.startswith("NAME")]
    except subprocess.CalledProcessError:
        return []

def is_model_installed(model_name):
    return model_name in get_installed_models()

def handle_model_selection(model):
    if not is_model_installed(model):
        try:
            subprocess.run(["ollama", "pull", model], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors du téléchargement du modèle {model}: {e}")

def get_available_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    return [voice.name for voice in voices]

def set_audio_params(engine, volume=1.0, rate=200, voice_name='Hortense'):
    engine.setProperty('volume', volume / 100.0)
    engine.setProperty('rate', rate)
    for voice in engine.getProperty('voices'):
        if voice_name.lower() in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break

def execute_command(command: str):
    for cmd, action in COMMANDS.items():
        if cmd in command:
            try:
                result, message = action()
                return message
            except Exception as e:
                return f"Erreur lors de l'exécution de la commande : {str(e)}"
    return None  # Commande non reconnue
def record_and_send(model, language="fr-FR", mic_index=0, voice_name='Hortense', volume=50, rate=200, history=[], stop_signal=None):
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()
    set_audio_params(engine, volume=volume, rate=rate, voice_name=voice_name)
    microphone = sr.Microphone(device_index=mic_index)
    in_ollama_mode = False

    while not stop_signal[0]:
        with microphone as source:
            print("Enregistrement en cours... Parlez maintenant.")
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                print("Aucun son détecté, réessai...")
                continue
        try:
            print("Analyse de la commande vocale...")
            user_input = recognizer.recognize_google(audio, language=language).lower()
            print(f"Vous avez dit : {user_input}")

            if 'mode au lama' in user_input:
                response = "Passage en mode au Lama, quelle est votre question?"
                in_ollama_mode = True
            elif in_ollama_mode:
                if 'mode commande' in user_input:
                    response = "Fermeture du mode au Lama. Retour en mode commande."
                    in_ollama_mode = False
                else:
                    response = generate_response(user_input, history, model)
                    response += "\nAvez-vous une autre question?"
            else:
                command_response = execute_command(user_input)
                if command_response:
                    response = command_response
                else:
                    response = "Désolé, je n'ai pas compris la commande. Pouvez-vous répéter?"

            print(f"Réponse : {response}")
            engine.say(response)
            engine.runAndWait()

        except sr.UnknownValueError:
            print("Erreur : La reconnaissance vocale n'a pas compris l'audio.")
        except sr.RequestError as e:
            print(f"Erreur : Impossible de contacter le service de reconnaissance vocale ; {e}")
        except Exception as e:
            print(f"Erreur inattendue : {str(e)}")

def launch_interface():
    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column():
                model_selection = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    label="💻 Sélection du modèle IA",
                    value="llama3.1" if "llama3.1" in AVAILABLE_MODELS else AVAILABLE_MODELS[0]
                )
            with gr.Column():
                with gr.Accordion(" 🎛️ Paramètres avancés", open=False):
                    gr.Markdown("🔩 Réglages des options avec Ollama 🦙")
                    mic_list = sr.Microphone.list_microphone_names()
                    mic_choices = [f"{i} {name}" for i, name in enumerate(mic_list)]
                    mic_index = gr.Dropdown(
                        choices=mic_choices,
                        label="🎙️ Sélection du Microphone",
                        value=mic_choices[0] if mic_choices else "0"
                    )
                    language = gr.Dropdown(['fr-FR', 'en-EN'], label="🌐 Langue", value='fr-FR')
                    available_voices = get_available_voices()
                    voice = gr.Dropdown(available_voices, label="🗣️ Voix", value='Hortense' if 'Hortense' in available_voices else available_voices[0])
                    volume = gr.Slider(minimum=0, maximum=100, step=10, label="🎚️ Volume", value=50)
                    rate = gr.Slider(50, 300, label="⏩ Vitesse de la voix", value=200)
                    
                output = gr.Markdown("## Etat du système de commandes", elem_classes="centered")

        with gr.Row():
            start_button = gr.Button("▶️ Commencer à parler")
            stop_button = gr.Button("⏹️ Arrêter l'enregistrement")

        stop_signal = [False]

        def toggle_recording(model_selection, language, mic_index, voice, volume, rate):
            stop_signal[0] = False
            mic_idx = int(mic_index.split()[0])
            threading.Thread(target=record_and_send, args=(
                model_selection, language, mic_idx, voice, volume, rate, [], stop_signal
            )).start()
            return "## 🔊 Enregistrement en cours..."

        def stop_recording():
            stop_signal[0] = True
            return "## 🔇 Enregistrement arrêté."

        start_button.click(
            toggle_recording,
            inputs=[model_selection, language, mic_index, voice, volume, rate],
            outputs=[output]
        )
        stop_button.click(stop_recording, inputs=[], outputs=[output])
        gr.Markdown()
        gr.Markdown("# Liste des commandes vocales")
        gr.Markdown()
        gr.Markdown("Pour passer en mode Ollama et discuter : 📢 Passage en Mode Ollama")
        gr.Markdown("Pour quitter le mode Ollama et retourner en mode commandse vocales : 📢 Retour en mode commande")
        gr.Markdown()
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Commandes des applications**")
                gr.Markdown("1. 📢 Ouvre la calculette\n2. 📢 Ouvre le bloc-notes\n3. 📢 Ouvre outlook\n4. 📢 Mets de la musique\n5. \n6. \n7. \n8.")
            with gr.Column():
                gr.Markdown("**Commandes du système Windows**")
                gr.Markdown("1. 📢 Ouvre le gestionnaire des tâches\n2. 📢 Ouvre le panneau des configurations\n3. 📢 Verrouille la session\n4. 📢 Ouvre les téléchargements\n5. 📢 Ouvre le dossier des documents\n6. 📢 Ouvre le dossier des images\n7. 📢 Ouvre le dossier vidéo\n8. 📢 Ouvre le dossier utilisateur")
            with gr.Column():
                gr.Markdown("**Commandes du navigateur Web**")
                gr.Markdown("1. 📢 Ouvre la mosaïque\n2. 📢 Ouvre Youtube\n3. 📢 Ouvre Facebook\n4. 📢 Ouvre Plex\n5. 📢 Ouvre le portail web\n6. \n7. \n8.")

    return interface

if __name__ == "__main__":
    launch_interface().launch(share=True)
