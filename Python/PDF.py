import os
import subprocess
import requests
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr
import numpy as np
import json
import logging
import urllib3
from typing import List, Tuple  # Ajouté pour les annotations de type

from ollama import Client  # Assurez-vous que la bibliothèque Ollama est installée

logging.basicConfig(level=logging.ERROR)
logging.getLogger("gradio").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Liste des modèles disponibles au téléchargement
available_models = [
    "llama3.1",
    "mistral",
    "starling-lm"
]

# Liste des prompts système
system_prompts = [
    "",
    "Dans ce formulaire administratif, quels sont les champs obligatoires à remplir dans ce formulaire administratif, quelles informations précises sont demandées, et quelles sont les éventuelles erreurs ou omissions qui pourraient entraîner un refus ou un retard dans le traitement de ce document officiel ?",

    "Dans ce livre d'auteur, peux-tu résumer les idées principales, thèmes abordés et le message global que l'auteur cherche à transmettre dans ce livre, en soulignant également les passages clés ou les moments forts qui marquent la progression de l’intrigue ou de l’argumentation ?",

    "Dans ce magazine de presse People, quels sujets, célébrités et événements sont mis en avant dans ce numéro de magazine People, et comment ces articles couvrent-ils des aspects de la vie personnelle, professionnelle ou publique de ces personnalités connues, avec des anecdotes ou détails particuliers ?",

    "Dans cette documentation technique, quelles sont les étapes précises pour installer, configurer ou dépanner le produit ou service décrit dans cette documentation technique, et quelles recommandations ou précautions doivent être suivies pour assurer un bon fonctionnement et éviter les erreurs fréquentes ?",

    "Dans cet acte notarié, quels sont les termes légaux, les clauses et obligations majeures énoncées dans cet acte notarié, et quels droits, responsabilités ou protections spécifiques cet acte confère-t-il aux parties concernées dans le cadre d’un contrat ou d’une transaction juridique ?",

    "Dans ces élément de justice, quels sont les faits, les preuves présentées et les arguments des parties dans cette affaire judiciaire, et quelles décisions ou conclusions ont été rendues par le tribunal, ainsi que les implications juridiques ou pénales de ce jugement pour les personnes impliquées ?",

    "Dans ce capture d'écra, quels éléments d'interface, informations ou détails visuels sont présents dans cette capture d'écran, et comment ces éléments peuvent-ils être interprétés pour diagnostiquer un problème technique, analyser une situation ou effectuer une action spécifique ?"
]

# Fonction pour lister les modèles disponibles localement via la commande 'ollama list'
def get_installed_models() -> List[str]:
    try:
        models = subprocess.check_output(["ollama", "list"], stderr=subprocess.DEVNULL).decode("utf-8")
        # Filtrer les modèles à exclure comme "failed" ou "aya"
        filtered_models = [
            model.split(":")[0].strip() for model in models.splitlines()
            if not (model.startswith("NAME") or "failed" in model or "aya" in model or "nomic-embed-text" in model or "mxbai-embed-large" in model or "bakllava" in model or "llava" in model or "llava-llama3" in model or "llava-phi3" in model or "minicpm-v" in model)
        ]
        return sorted(filtered_models)
    except subprocess.CalledProcessError as e:
        logging.error(f"Erreur lors de l'exécution de la commande 'ollama list': {e}")
        return []

# Fonction combinant les modèles installés et disponibles au téléchargement
def get_models() -> List[str]:
    installed_models = get_installed_models()
    all_models = sorted(set(installed_models + available_models))
    return all_models

# Fonction pour vérifier si un modèle est installé
def is_model_installed(model_name: str) -> bool:
    installed_models = get_installed_models()
    return model_name in installed_models

# Fonction pour gérer la sélection et le téléchargement des modèles
def handle_model_selection(model: str) -> None:
    if not is_model_installed(model):
        logging.info(f"Le modèle {model} n'est pas trouvé. Tentative de téléchargement...")
        try:
            subprocess.run(["ollama", "pull", model], check=True, text=True)
            logging.info(f"Le modèle {model} a été téléchargé et installé avec succès.")
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.strip() if e.stderr else "Une erreur est survenue lors du téléchargement du modèle."
            logging.error(f"Erreur lors du téléchargement du modèle {model} : {error_message}")
            raise Exception(f"Échec du téléchargement du modèle {model}. Veuillez essayer de le télécharger manuellement.")
    else:
        logging.info(f"Le modèle {model} est déjà installé.")

# Fonction pour formater l'historique de chat (utile si vous souhaitez conserver un historique dans le chatbot)
def format_history(msg: str, history: List[Tuple[str, str]], system_prompt: str):
    chat_history = [{"role": "system", "content": system_prompt}]
    for query, response in history:
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})
    chat_history.append({"role": "user", "content": msg})
    return chat_history

# Initialisation du modèle de vectorisation avec Sentence Transformers
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("Modèle SentenceTransformer chargé avec succès")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle SentenceTransformer : {str(e)}")
    embedding_model = None  # Assurez-vous de gérer ce cas dans les fonctions suivantes

# Fonction pour vectoriser le texte
def vectorize_text(texts: List[str]) -> np.ndarray:
    try:
        logging.info("Vectorisation du texte...")
        vectors = embedding_model.encode(texts, convert_to_numpy=True)
        return vectors
    except Exception as e:
        logging.error(f"Erreur lors de la vectorisation : {str(e)}")
        return np.array([])  # Retourne un array vide en cas d'erreur

# Création de l'index FAISS
def create_faiss_index(vectors: np.ndarray):
    try:
        d = vectors.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(vectors)
        logging.info("Index FAISS créé avec succès")
        return index
    except Exception as e:
        logging.error(f"Erreur lors de la création de l'index FAISS : {str(e)}")
        return None

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() for page in reader.pages if page.extract_text() is not None)
        logging.info("Extraction du texte réussie")
        return text
    except Exception as e:
        logging.error(f"Erreur lors de l'extraction du texte PDF : {str(e)}")
        return f"Erreur lors de la lecture du PDF : {str(e)}"

# Fonction pour interroger l'API d'Ollama avec le modèle et le prompt sélectionnés
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def query_ollama(prompt: str, context: str, temperature: float, top_k: int, top_p: float, 
                presence_penalty: float, frequency_penalty: float, repeat_penalty: float, 
                num_keep: int, num_predict: int, num_ctx: int, model_selection: str, 
                system_prompt: str) -> str:
    try:
        logging.info("Interrogation de l'API Ollama...")

        final_prompt = f"{system_prompt}\n\n### Contexte :\n{context}\n\n### Question :\n{prompt}"

        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": model_selection,
            "prompt": final_prompt,
            "options": {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "repeat_penalty": repeat_penalty,
                "num_keep": num_keep,
                "num_predict": num_predict,
                "num_ctx": num_ctx
            }
        }
        json_data = json.dumps(data, ensure_ascii=False)
        logging.info(f"JSON data sent to Ollama API: {json_data}")

        response = requests.post(OLLAMA_API_URL, data=json_data, headers=headers, timeout=10)
        response.raise_for_status()

        response_content = response.content.decode('utf-8', errors='ignore').strip()
        logging.info(f"Raw response (limited to 500 chars): {response_content[:500]}")

        # Extraction des fragments JSON
        import re
        json_fragments = re.findall(r'({.*?})', response_content)

        final_response = ""
        done = False
        for fragment in json_fragments:
            try:
                fragment_json = json.loads(fragment)
                final_response += fragment_json.get('response', "")
                if fragment_json.get('done', False):
                    done = True
                    break
            except json.JSONDecodeError as e:
                logging.error(f"JSONDecodeError while processing fragment: {fragment[:100]} - {str(e)}")
                continue

        if not done:
            logging.warning("Réponse incomplète, certains fragments peuvent manquer.")

        return final_response if final_response else "Aucune réponse générée"

    except requests.RequestException as e:
        logging.error(f"Erreur lors de l'interrogation de l'API Ollama : {str(e)}")
        return f"Erreur lors de l'interrogation de l'API Ollama : {str(e)}"

# Variables globales pour stocker les documents vectorisés et l'index FAISS
document_texts: List[str] = []
faiss_index = None

# Fonction pour ajouter un PDF et le vectoriser automatiquement
def add_pdf_and_vectorize(pdf) -> str:
    global document_texts, faiss_index
    try:
        text = extract_text_from_pdf(pdf.name)
        if isinstance(text, str) and text.startswith("Erreur"):
            return text
        document_texts.append(text)
        vector = vectorize_text([text])
        if vector.size == 0:
            return "Erreur lors de la vectorisation du document"

        vector = np.array(vector)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        if faiss_index is None:
            faiss_index = create_faiss_index(vector)
            if faiss_index is None:
                return "Erreur lors de la création de l'index FAISS"
        else:
            faiss_index.add(vector)

        logging.info(f"Document {pdf.name} ajouté et vectorisé.")
        return f"Document {pdf.name} ajouté et vectorisé."
    except Exception as e:
        logging.error(f"Erreur lors de l'ajout du PDF : {str(e)}")
        return f"Erreur lors de l'ajout du PDF : {str(e)}"

# Fonction pour répondre à une question
def answer_question(question: str, temperature: float, top_k: int, top_p: float, 
                   presence_penalty: float, frequency_penalty: float, repeat_penalty: float, 
                   num_keep: int, num_predict: int, num_ctx: int, model_selection: str, 
                   system_prompt: str) -> str:
    global faiss_index, document_texts
    try:
        if faiss_index is None or not document_texts:
            return "Aucun document ajouté."

        question_vector = vectorize_text([question])
        if question_vector.size == 0:
            return "Erreur lors de la vectorisation de la question"
        question_vector = question_vector[0].reshape(1, -1)

        D, I = faiss_index.search(question_vector, k=1)
        if I[0][0] == -1:
            return "Aucun contexte trouvé pour la question posée."
        context = document_texts[I[0][0]]

        return query_ollama(question, context, temperature, top_k, top_p, presence_penalty, 
                            frequency_penalty, repeat_penalty, num_keep, num_predict, 
                            num_ctx, model_selection, system_prompt)
    except Exception as e:
        logging.error(f"Erreur lors de la réponse à la question : {str(e)}")
        return f"Erreur lors de la réponse à la question : {str(e)}"

# Interface Gradio
def launch_interface():
    try:
        models = get_models()

        with gr.Blocks() as interface:
            with gr.Row():
                with gr.Column(scale=1):
                    model_selection = gr.Dropdown(
                        choices=models,
                        label="💻 Sélection du modèle IA",
                        value="llama3.1" if "llama3.1" in models else models[0],
                        interactive=True,
                        elem_id="model-selection"
                    )
                with gr.Column(scale=3):
                    system_prompt = gr.Dropdown(
                        choices=system_prompts,
                        label="📃 Sélection du Prompt Système",
                        value=system_prompts[0],
                        interactive=True
                    )

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
                            minimum=1024, maximum=8192, step=1024, label="🎚️ Num Context", value=7168,
                            info="Taille maximale du contexte pris en compte pour chaque génération, influençant la pertinence des réponses basées sur l'entrée fournie."
                        )


            with gr.Row():
                with gr.Column(scale=1):
                    pdf_input = gr.File(label="Ajoutez un fichier PDF", type="filepath")
                    output_add_pdf = gr.Textbox(label="📦 Vecteurs du PDF", interactive=False)
                    question_input = gr.Textbox(
                        lines=7, 
                        label="❓ Posez votre question", 
                        placeholder="Entrez une question ici..."
                    )
                with gr.Column(scale=2):
                    answer_output = gr.Textbox(lines=20, label="📝 Réponse", interactive=False, elem_id="Réponse")
                    submit_question_button = gr.Button("🗨️ Envoyer")
                    
            # Actions liées aux interactions
            pdf_input.change(fn=add_pdf_and_vectorize, inputs=pdf_input, outputs=output_add_pdf)
            submit_question_button.click(
                fn=answer_question,
                inputs=[
                    question_input, temperature, top_k, top_p, presence_penalty, 
                    frequency_penalty, repeat_penalty, num_keep, num_predict, 
                    num_ctx, model_selection, system_prompt
                ],
                outputs=answer_output
            )

            # Action lors de la sélection d'un modèle : Vérifier et télécharger si nécessaire
            model_selection.change(
                fn=lambda model: handle_model_selection(model) or gr.update(),
                inputs=model_selection,
                outputs=None,
                queue=False  # Exécuter immédiatement sans file d'attente
            )

        logging.info("Interface Gradio lancée avec succès.")
        return interface
    except Exception as e:
        logging.error(f"Erreur lors du lancement de l'interface : {str(e)}")
        raise

if __name__ == "__main__":
    interface = launch_interface()
    interface.launch(share=True)
