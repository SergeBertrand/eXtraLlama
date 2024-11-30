import os
import sys
import subprocess
import gradio as gr
import ollama
from typing import List, Dict  # Importer List et Dict pour les annotations de type

# Liste des modèles disponibles pour l'utilisateur
available_models = ["llama3.2-vision", "minicpm-v", "llava", "bakllava", "llava-llama3", "llava-phi3", "llama3.2"]

# Fonction pour récupérer les modèles installés, en filtrant les erreurs de console
def get_installed_models():
    try:
        models = subprocess.check_output(["ollama", "list"], stderr=subprocess.DEVNULL).decode("utf-8")
        # Filtrer les modèles à exclure comme "failed" ou "aya"
        filtered_models = [
            model.split(":")[0].strip() for model in models.splitlines()
            if not (model.startswith("NAME") or "failed" in model or "aya" in model or "nomic-embed-text" in model or "mxbai-embed-large")
        ]
        return sorted(set(filtered_models))  # Supprime les doublons
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de la commande 'ollama list': {e}")
        return []

# Vérifier si le modèle est installé
def is_model_installed(model_name):
    installed_models = get_installed_models()
    return model_name in installed_models

# Gestion de la sélection du modèle, avec téléchargement si nécessaire
def handle_model_selection(model):
    if not is_model_installed(model):
        print(f"Le modèle {model} n'est pas trouvé. Tentative de téléchargement...")
        try:
            subprocess.run(["ollama", "pull", model], check=True, text=True)
            print(f"Le modèle {model} a été téléchargé et installé avec succès.")
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.strip() if e.stderr else "Une erreur est survenue lors du téléchargement du modèle."
            print(f"Erreur lors du téléchargement du modèle {model} : {error_message}")
    else:
        print(f"Le modèle {model} est déjà installé.")

# Liste des prompts système
system_prompts = [
    "",
    "Quel est l’élément central de cette image ? Décris précisément ce qui semble être l’objet, la personne ou la scène principale ? Réponds toujours en français. ",
    "Quels indices dans l’arrière-plan ou les détails secondaires de cette image fournissent des informations supplémentaires sur son origine ou son contexte ? Réponds toujours en français.",
    "Quelle émotion ou ambiance générale l’image essaie-t-elle de transmettre, en termes de lumière, de couleurs et de composition artistique ? Réponds toujours en français.",
    "Qui sont les personnages présents dans l’image ? Que peuvent révéler leurs expressions faciales, vêtements ou postures sur leur histoire ou leur rôle ? Réponds toujours en français.",
    "Cette image semble-t-elle être une photo, un dessin, une peinture, ou un autre type de représentation visuelle ? Réponds toujours en français."
]

# Génération de la réponse
def generate_response(msg: str, history: List[Dict[str, str]], system_prompt: str, model: str, image_input=None):
    handle_model_selection(model)  # Vérification et téléchargement du modèle si nécessaire

    # Construire le contexte de conversation à partir de l'historique
    conversation = ''
    if system_prompt:
        conversation += system_prompt + '\n\n'
    for message in history:
        role = message['role']
        content = message['content']
        if role == 'user':
            conversation += f'Utilisateur : {content}\n'
        elif role == 'assistant':
            conversation += f'Assistant : {content}\n'
    conversation += f'Utilisateur : {msg}\nAssistant :'

    try:
        response_generator = ollama.generate(
            model=model,
            prompt=conversation,
            images=[image_input],
            stream=True
        )
        response = ""
        for partial_resp in response_generator:
            token = partial_resp['response']
            response += token
            yield response
    except Exception as e:
        yield f"Erreur lors de la génération de la réponse : {str(e)}"

# Interface Gradio
def launch_interface():
    with gr.Blocks() as interface:
        gr.Markdown("## Interagir avec Ollama")
        with gr.Row():
            with gr.Column(scale=1):
                model_selection = gr.Dropdown(
                    choices=available_models,  # Limite aux modèles spécifiés
                    label="💻 Sélection du modèle IA",
                    value=available_models[0] if available_models else None,  # Sélectionne par défaut le premier modèle de la liste
                )
                
            with gr.Column(scale=4):
                system_prompt = gr.Dropdown(
                    choices=system_prompts,
                    label="📜 Sélection du Prompt System",
                    value=system_prompts[0]
                )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Chargez une image")
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label="Chatbot Vision IA", height=500, type="messages")

        # Zone de Saisie et Boutons
        with gr.Row():
            user_input = gr.Textbox(
                label="Votre message",
                placeholder="Tapez votre message ici...",
                lines=1,
                interactive=True
            )
        with gr.Row():
            undo_btn = gr.Button("↩️ Supprimer la réponse")
            clear_btn = gr.Button("🛢️ Supprimer la discussion")
            retry_btn = gr.Button("🔄 Nouvelle réponse")
            submit_btn = gr.Button("💬 Envoyer")

        # État pour l'historique
        history = gr.State([])

        # Fonctions pour gérer les interactions des boutons
        def send_message(user_message, history, system_prompt, model, image_input):
            if not user_message.strip():
                return history, "", history  # Ne rien faire si le message est vide

            response_generator = generate_response(user_message, history, system_prompt, model, image_input)
            history.append({"role": "user", "content": user_message})
            response = ""
            for partial_response in response_generator:
                response = partial_response  # Mettre à jour la réponse à chaque étape
                yield history + [{"role": "assistant", "content": response}], "", history + [{"role": "assistant", "content": response}]

            history.append({"role": "assistant", "content": response})
            yield history, "", history

        def clear_chat():
            return [], "", []  # Réinitialiser l'historique, vider la saisie, vider le chatbot

        def undo_last_response(current_history):
            if current_history:
                current_history.pop()  # Supprimer la dernière interaction (user, bot)
                if current_history:
                    current_history.pop()  # Supprimer la dernière interaction (bot)
            return current_history, "", current_history

        def retry_response(history, system_prompt, model_selection, image_input):
            if history:
                last_user_message = history[-2]['content'] if len(history) >= 2 else ""
                # Réessayer de générer la réponse pour le dernier message utilisateur
                response_generator = generate_response(last_user_message, history[:-2], system_prompt, model_selection, image_input)
                history.append({"role": "user", "content": last_user_message})
                response = ""
                for partial_response in response_generator:
                    response = partial_response  # Mettre à jour la réponse à chaque étape
                    yield history + [{"role": "assistant", "content": response}], "", history + [{"role": "assistant", "content": response}]

                history.append({"role": "assistant", "content": response})
                yield history, "", history

        # Lier les boutons aux fonctions
        submit_btn.click(
            fn=send_message,
            inputs=[user_input, history, system_prompt, model_selection, image_input],
            outputs=[history, user_input, chatbot]
        )

        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[history, user_input, chatbot]
        )

        undo_btn.click(
            fn=undo_last_response,
            inputs=[history],
            outputs=[history, user_input, chatbot]
        )

        retry_btn.click(
            fn=retry_response,
            inputs=[history, system_prompt, model_selection, image_input],
            outputs=[history, user_input, chatbot]
        )

    return interface

if __name__ == "__main__":
    print("coucou")
    print(available_models)  # Afficher la liste des modèles disponibles au lancement
    print("coucou")
    launch_interface().launch()
