import gradio as gr
import pyttsx3
import requests
import subprocess
import json
from typing import List, Tuple
from transformers import pipeline
import numpy as np

# Initialisation du pipeline ASR de Transformers avec Whisper petite taille (multilingue)
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# Liste des modèles disponibles
available_models = [
    "llama3.1", 
    "mistral",
    "starling-lm"
]

# Liste des prompts système
system_prompts = [
    "Réponds toujours en français.",    
    "Always answer in English.",
    "Antworte immer auf Deutsch.",
    "Responda siempre en español.",
    "Rispondete sempre in italiano."
]

# Fonction pour formater l'historique des conversations
def format_history(msg: str, history: List[Tuple[str, str]], system_prompt: str):
    chat_history = [{"role": "system", "content": system_prompt}]
    for query, response in history:
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})
    chat_history.append({"role": "user", "content": msg})
    return chat_history

# Fonction pour filtrer les répétitions dans la réponse générée
def remove_repeated_phrases(text: str) -> str:
    phrases = text.split()
    seen = set()
    result = []
    for phrase in phrases:
        if phrase not in seen:
            result.append(phrase)
            seen.add(phrase)
    return ' '.join(result)

# Fonction pour envoyer une requête au modèle avec gestion d'erreurs
def generate_response(msg: str, history: List[Tuple[str, str]], system_prompt: str, model: str, 
                     temperature=0.8, num_predict=1024, top_k=40, top_p=0.92, 
                     presence_penalty=0.5, repeat_penalty=1.5, frequency_penalty=1.8, num_ctx=7168, num_keep=3072):
    
    chat_history = format_history(msg, history, system_prompt)

    # Vérifie si le modèle est installé, sinon tente de le télécharger
    if not is_model_installed(model):
        handle_model_selection(model)
        if not is_model_installed(model):
            return f"Erreur : Le modèle {model} n'a pas pu être téléchargé."

    try:
        # Utilisation du client Ollama pour générer une réponse
        response_generator = requests.post(
            f"http://localhost:11434/api/chat",
            json={"model": model, "messages": chat_history, "options": {
                "temperature": temperature, "top_k": top_k, "top_p": top_p,
                "presence_penalty": presence_penalty, "repeat_penalty": repeat_penalty,
                "frequency_penalty": frequency_penalty, "num_predict": num_predict,
                "num_ctx": num_ctx, "num_keep": num_keep
            }},
            stream=True,
            timeout=60  # Ajustez le timeout si nécessaire
        )

        response_generator.raise_for_status()

        # Accumuler la réponse complète depuis le générateur
        message = ""
        for line in response_generator.iter_lines():
            if line:
                try:
                    content = json.loads(line).get('message', {}).get('content', '')
                    if content:
                        message += content
                except json.JSONDecodeError:
                    print(f"Erreur de décodage JSON pour la ligne: {line}")
        
        # Supprimer les répétitions dans la réponse
        return remove_repeated_phrases(message)

    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête à l'API Ollama: {e}")
        return f"Erreur lors de la génération de la réponse : {e}"
    except Exception as e:
        print(f"Erreur inattendue dans generate_response: {e}")
        return f"Erreur inattendue : {e}"

# Vérification de l'installation des modèles
def get_installed_models():
    try:
        models = subprocess.check_output(["ollama", "list"], stderr=subprocess.DEVNULL).decode("utf-8")
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

# Fonction pour récupérer la liste des voix disponibles
def get_available_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    return [voice.name for voice in voices]

# Configuration des paramètres audio et de la synthèse vocale
def set_audio_params(volume=1.0, rate=200, voice_name='Hortense'):
    engine = pyttsx3.init()
    engine.setProperty('volume', volume)
    engine.setProperty('rate', rate)
    voices = engine.getProperty('voices')
    for v in voices:
        if voice_name.lower() in v.name.lower():
            engine.setProperty('voice', v.id)
            print(f"Voix sélectionnée : {v.name}")
            break
    else:
        print("Erreur : La voix spécifiée n'a pas été trouvée.")
    return engine

# Fonction pour transcrire l'audio en texte en utilisant le pipeline ASR
def transcribe_audio(audio):
    if audio is None:
        return "Erreur : Aucun audio fourni."
    
    try:
        sr, y = audio

        # Convertir en mono si stéréo
        if y.ndim > 1:
            y = y.mean(axis=1)
            
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))

        transcription = transcriber({"sampling_rate": sr, "raw": y})["text"]
        return transcription
    except Exception as e:
        print(f"Erreur lors de la transcription de l'audio : {e}")
        return f"Erreur lors de la transcription de l'audio : {e}"

# Fonction pour générer une réponse à partir de l'audio et la synthèse vocale
def process_audio(audio, model_selection, system_prompt, voice, volume, rate, history):
    # Transcrire l'audio
    user_input = transcribe_audio(audio)
    if "Erreur" in user_input:
        return user_input
    
    print(f"Vous avez dit : {user_input}")

    # Retourner le transcript pour affichage
    return user_input

# Fonction pour envoyer le texte modifié au modèle et générer une réponse
def send_text(user_input, model_selection, system_prompt, voice, volume, rate, history_state):
    if not user_input.strip():
        return "Erreur : Le texte est vide.", history_state

    # Générer une réponse
    response = generate_response(user_input, history_state, system_prompt, model_selection)

    # Synthèse de la réponse vocale avec pyttsx3
    try:
        engine = set_audio_params(volume / 100, rate, voice)
        engine.say(response)
        engine.runAndWait()
    except Exception as e:
        print(f"Erreur lors de la synthèse vocale : {e}")

    # Mettre à jour l'historique
    history_state.append((user_input, response))
    return response, history_state

# Interface Gradio avec la gestion des prompts, modèles et historique
def launch_interface():
    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column(scale=1):
                # Dropdown pour sélectionner le modèle IA
                model_selection = gr.Dropdown(
                    choices=available_models,
                    label="💻 Sélection du modèle IA",
                    value="llama3.1" if "llama3.1" in available_models else available_models[0]
                )
            with gr.Column(scale=3):
                # Sélection du prompt système
                system_prompt = gr.Dropdown(
                    choices=system_prompts,
                    label="📃 Sélection du Prompt Système",
                    value=system_prompts[0]
                )


        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion(" 🎛️ Paramètres avancés", open=False):
                    with gr.Column():

                        # Liste des voix disponibles
                        available_voices = get_available_voices()
                        voice = gr.Dropdown(available_voices, label="🔊 Voix", value='Hortense' if 'Hortense' in available_voices else available_voices[0])

                        # Paramètres de la voix
                        volume = gr.Slider(minimum=0, maximum=100, step=10, label="🎚️ Volume", value=50)
                        rate = gr.Slider(50, 300, label="⏩ Vitesse de la voix", value=200)

                # Composant audio pour enregistrer la voix directement depuis le microphone
                audio_input = gr.Audio(type="numpy", label="🎤 Enregistrer votre voix")
                
                # Bouton pour transcrire l'audio
                transcribe_button = gr.Button("🔄 Transcrire")

                # Bouton pour envoyer le transcript modifié
                send_button = gr.Button("👉 Envoyer")

            with gr.Column(scale=2):
                # Zone de texte pour afficher et modifier le transcript
                transcript = gr.Textbox(label="📝 Transcript", lines=4, placeholder="Le transcript apparaîtra ici...", interactive=True)

                # Zone de texte pour afficher la réponse
                output = gr.Textbox(label="💬 Réponse de l'assistant", elem_id="Réponse", lines=10)
                


        # Historique des conversations (optionnel)
        history = gr.State([])

        # Associer le bouton de transcription à la fonction de transcription
        transcribe_button.click(
            process_audio, 
            inputs=[audio_input, model_selection, system_prompt, voice, volume, rate, history],
            outputs=transcript
        )

        # Associer le bouton d'envoi à la fonction d'envoi
        send_button.click(
            send_text,
            inputs=[transcript, model_selection, system_prompt, voice, volume, rate, history],
            outputs=[output, history]
        )

    return interface

# Exécution de l'interface
if __name__ == "__main__":
    try:
        launch_interface().launch()
    except Exception as e:
        print(f"Erreur lors du lancement de l'interface : {e}")
