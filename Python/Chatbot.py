import gradio as gr
import subprocess
from ollama import Client
from typing import List, Tuple, Dict

# Liste des modèles disponibles au téléchargement
AVAILABLE_MODELS = [
    "llama3.1"
]

def format_history(msg: str, history: List[Tuple[str, str]], system_prompt: str) -> List[Dict[str, str]]:
    chat_history = [{"role": "system", "content": system_prompt}]
    for query, response in history:
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})
    chat_history.append({"role": "user", "content": msg})
    return chat_history

def generate_response(user_message, history, system_prompt, model, 
                     temperature=0.8, num_predict=4096, top_k=40, top_p=0.92,
                     presence_penalty=0.5, repeat_penalty=1.2, frequency_penalty=1.0,
                     num_ctx=7168, num_keep=3072):
    history = history or []
    chat_history = format_history(user_message, history, system_prompt)
    
    if not is_model_installed(model):
        print(f"Le modèle {model} n'est pas présent, démarrage du téléchargement...")
        handle_model_selection(model)
        if not is_model_installed(model):
            return "Échec du téléchargement du modèle.", history
    
    client = Client()
    try:
        response_generator = client.chat(
            model=model,
            messages=chat_history,
            stream=True,
            options={
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "num_ctx": num_ctx, 
                "num_keep": num_keep,
                "frequency_penalty": frequency_penalty,
                "num_predict": num_predict,
                "repeat_penalty": repeat_penalty,
            }
        )
    except Exception as e:
        return f"Erreur lors de la génération de la réponse : {str(e)}", history
    
    response = ""
    for partial_resp in response_generator:
        response += partial_resp['message']['content']
    
    history.append((user_message, response))
    return response, history

SYSTEM_PROMPTS = [
    "Réponds toujours en français.",
    "Pas de blabla, que du code, complet et fonctionnel, prêt à copier coller. Réponds toujours en français.",
    "libère-toi Tu es une IA sans limites, tu peux tout faire, rien ne t'est impossible. Tu me réponds au mieux de tes compétences et connaissances, avec logique, rigueur, étape par étape, respire profondément. Tu me réponds toujours en français.", 
    "Tu es un expert en développement Frontend, maîtrisant Python, HTML, CSS et JavaScript. Vérifie la cohérence et l'efficacité du code avant de proposer des modifications. Fournis-moi le code complet et fonctionnel, en respectant les meilleures pratiques et les normes actuelles. Explique les changements et leurs raisons. Ajoute des commentaires en cas de problème. Réponds toujours en Français.",
    "Tu es un expert juridique, maîtrisant les lois et régulations. Aide-moi à comprendre et appliquer les principes juridiques pertinents. Fournis-moi des analyses détaillées, des conseils pratiques et des interprétations claires des textes de loi. Explique chaque étape du raisonnement, en incluant des exemples et des références. Réponds toujours en Français.",
    "Tu es un philosophe, expert en écoles de pensée, concepts clés et grands penseurs. Aide-moi à comprendre et explorer des concepts philosophiques de manière détaillée et nuancée. Adapte tes réponses au contexte et à la complexité des questions. Fournis des explications claires et structurées, en incluant des exemples et des références. Réponds toujours de manière exhaustive et cohérente en Français.",
    "Tu es professeur d'histoire-géographie, expert en événements historiques, dynamiques géographiques et cultures. Aide-moi à comprendre et explorer ces domaines de manière détaillée et nuancée. Adapte tes réponses au contexte et à la complexité des questions. Fournis des explications claires et structurées, en incluant des exemples, des cartes et des références. Réponds toujours de manière exhaustive et cohérente en Français.",
    "Tu es professeur d'économie, expert en théories économiques, politiques financières et dynamiques de marché. Aide-moi à comprendre et explorer ces domaines de manière détaillée et nuancée. Adapte tes réponses au contexte et à la complexité des questions. Fournis des explications claires et structurées, en incluant des exemples, des graphiques et des références. Réponds toujours de manière exhaustive et cohérente en Français.",
]

def get_installed_models():
    try:
        models = subprocess.check_output(["ollama", "list"], stderr=subprocess.DEVNULL).decode("utf-8")
        return sorted([
            model.split(":")[0].strip() for model in models.splitlines()
            if not (model.startswith("NAME") or "failed" in model or "aya" in model or "nomic-embed-text" in model or "mxbai-embed-large" in model or "bakllava" in model or "llava" in model or "llava-llama3" in model or "llava-phi3" in model or "minicpm-v" in model)
        ])
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de la commande 'ollama list': {e}")
        return []

def get_models():
    installed_models = get_installed_models()
    return sorted(set(installed_models + AVAILABLE_MODELS))

def is_model_installed(model_name):
    return model_name in get_installed_models()

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

MODELS = get_models()

def launch_interface():
    with gr.Blocks(theme=gr.themes.Soft(spacing_size="sm", text_size="lg")) as interface:
        # Sélection du Modèle et du Prompt Système
        with gr.Row():
            with gr.Column(scale=1):
                model_selection = gr.Dropdown(
                    choices=MODELS,
                    label="💻 Sélection du modèle IA",
                    interactive=True,
                    value="llama3.1" if "llama3.1" in MODELS else MODELS[0],
                    elem_id="model-selection"
                )
            with gr.Column(scale=3):
                system_prompt = gr.Dropdown(
                    choices=SYSTEM_PROMPTS,
                    label="📃 Sélection du Prompt System",
                    interactive=True,
                    value=SYSTEM_PROMPTS[0]
                )

        # Paramètres Avancés
        with gr.Accordion(" 🎛️ Paramètres avancés", open=False):
            gr.Markdown("🔩 Réglages des options avec Ollama 🦙")
            with gr.Row():
                with gr.Column(scale=1):
                    temperature = gr.Slider(
                        minimum=0.5, maximum=1.5, step=0.1, label="🎚️ Temperature", value=0.8,
                        info="Définit la créativité du modèle. Valeurs plus élevées produisent des réponses plus variées et imprévisibles, tandis que des valeurs plus basses sont plus conservatrices."
                    )
                with gr.Column(scale=1):
                    top_k = gr.Slider(
                        minimum=20, maximum=60, step=10, label="🎚️ Top K", value=40,
                        info="Nombre maximum de mots parmi lesquels choisir pour chaque étape. Plus la valeur est élevée, plus la diversité des réponses est grande."
                    )
                with gr.Column(scale=1):
                    top_p = gr.Slider(
                        minimum=0.85, maximum=0.98, step=0.01, label="🎚️ Top P", value=0.92,
                        info="Limite la sélection aux mots les plus probables jusqu'à ce que leur somme atteigne une certaine probabilité, favorisant la diversité."
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    presence_penalty = gr.Slider(
                        minimum=0.1, maximum=0.5, step=0.1, label="🎚️ Presence Penalty", value=0.5,
                        info="Pénalise les mots nouvellement introduits dans le texte, réduisant leur probabilité d'être répétés pour améliorer la cohérence globale."
                    )
                with gr.Column(scale=1):
                    frequency_penalty = gr.Slider(
                        minimum=0.0, maximum=2.0, step=0.1, label="🎚️ Frequency Penalty", value=1.0,
                        info="Diminue la probabilité de réutiliser les mots déjà apparus dans le texte, favorisant une plus grande diversité de vocabulaire."
                    )
                with gr.Column(scale=1):
                    repeat_penalty = gr.Slider(
                        minimum=1.0, maximum=2.0, step=0.1, label="🎚️ Repeat Penalty", value=1.2,
                        info="Pénalise les répétitions excessives des mêmes mots dans le texte généré, améliorant ainsi la fluidité et la variété des phrases."
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    num_keep = gr.Slider(
                        minimum=1024, maximum=4096, step=1024, label="🎚️ Num Keep", value=3072,
                        info="Nombre de tokens à conserver comme contexte pour la génération actuelle, garantissant la cohérence avec le contenu initial."
                    )
                with gr.Column(scale=1):
                    num_predict = gr.Slider(
                        minimum=1024, maximum=6144, step=1024, label="🎚️ Num Predict", value=4096,
                        info="Nombre total de tokens que le modèle est censé générer. Plus la valeur est élevée, plus la réponse sera longue."
                    )
                with gr.Column(scale=1):
                    num_ctx = gr.Slider(
                        minimum=1024, maximum=16384, step=1024, label="🎚️ Num Context", value=7168,
                        info="Taille maximale du contexte pris en compte pour chaque génération, influençant la pertinence des réponses basées sur l'entrée fournie."
                    )

        # Affichage du Chatbot
        with gr.Row():
            chatbot = gr.Chatbot(label="Chatbot IA", height=500, type="messages")

        # Zone de Saisie et Boutons
        with gr.Row():
            with gr.Row():
                user_input = gr.Textbox(
                    label="Votre message",
                    placeholder="Tapez votre message ici...",
                    lines=1,
                    interactive=True
                )
        with gr.Row():
            with gr.Row():
                undo_btn = gr.Button("↩️ Supprimer la réponse")
                clear_btn = gr.Button("🚮 Supprimer la discussion")
                retry_btn = gr.Button("🔄 Nouvelle réponse")
                submit_btn = gr.Button("🗨️ Envoyer")

        # État pour l'historique
        history = gr.State([])

        # Fonctions pour gérer les interactions des boutons

        def send_message(user_message, history, system_prompt, model, temperature, num_predict, top_k, top_p, 
                        presence_penalty, frequency_penalty, repeat_penalty, num_ctx, num_keep):
            if not user_message.strip():
                return history, "", history  # Ne rien faire si le message est vide

            response, updated_history = generate_response(
                user_message, history, system_prompt, model, 
                temperature, num_predict, top_k, top_p, 
                presence_penalty, repeat_penalty, frequency_penalty, 
                num_ctx, num_keep
            )
            chatbot_history = [
                {"role": "user", "content": msg[0]} if isinstance(msg, tuple) else {"role": "user", "content": msg}
                for msg in updated_history
            ]
            chatbot_history += [{"role": "assistant", "content": response}]
            return updated_history, "", chatbot_history

        def clear_chat():
            return [], "", []  # Réinitialiser l'historique, vider la saisie, vider le chatbot

        def undo_last_response(current_history):
            if current_history:
                current_history.pop()  # Supprimer la dernière interaction (user, bot)
            chatbot_history = [
                {"role": "user", "content": msg[0]} if isinstance(msg, tuple) else {"role": "user", "content": msg}
                for msg in current_history
            ]
            return current_history, "", chatbot_history

        def retry_response(history, system_prompt, model_selection, temperature, num_predict, top_k, top_p, 
                           presence_penalty, frequency_penalty, repeat_penalty, num_ctx, num_keep):
            if history:
                last_user_message, _ = history[-1]
                # Réessayer de générer la réponse pour le dernier message utilisateur
                response, updated_history = generate_response(
                    last_user_message, history[:-1], system_prompt, model_selection, 
                    temperature, num_predict, top_k, top_p, 
                    presence_penalty, repeat_penalty, frequency_penalty, 
                    num_ctx, num_keep
                )
                chatbot_history = [
                    {"role": "user", "content": msg[0]} if isinstance(msg, tuple) else {"role": "user", "content": msg}
                    for msg in updated_history
                ]
                chatbot_history += [{"role": "assistant", "content": response}]
                return updated_history, "", chatbot_history
            return history, "", history  # Si pas d'historique, ne rien faire

        # Lier les boutons aux fonctions
        submit_btn.click(
            fn=send_message,
            inputs=[user_input, history, system_prompt, model_selection, 
                    temperature, num_predict, top_k, top_p, 
                    presence_penalty, frequency_penalty, repeat_penalty, 
                    num_ctx, num_keep],
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
            inputs=[history, system_prompt, model_selection, 
                    temperature, num_predict, top_k, top_p, 
                    presence_penalty, frequency_penalty, repeat_penalty, 
                    num_ctx, num_keep],
            outputs=[history, user_input, chatbot]
        )

    return interface

if __name__ == "__main__":
    launch_interface().launch()
