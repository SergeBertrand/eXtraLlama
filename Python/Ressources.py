import os
import subprocess
import requests
import gradio as gr
from tqdm import tqdm  # Import tqdm for terminal progress bar

# Base paths
base_path = os.path.expandvars(r"%userprofile%\eXtraLlama")
comfyUI_path = os.path.expandvars(r"%userprofile%\ComfyUI\ComfyUI")

UNET_DIR = os.path.join(comfyUI_path, "models", "unet")
CHECKPOINT_DIR = os.path.join(comfyUI_path, "models", "checkpoints")
VAE_DIR = os.path.join(comfyUI_path, "models", "vae")
CLIP_DIR = os.path.join(comfyUI_path, "models", "clip")
CLIP_VISION_DIR = os.path.join(comfyUI_path, "models", "clip_vision")
UPSCALE_DIR = os.path.join(comfyUI_path, "models", "upscale_models")

# URL dictionaries
UNET_URLS = {
    "": "",
    "flux1-dev.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/unet/flux1-dev.safetensors",
    "flux1-schnell.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/unet/flux1-schnell.safetensors",
    "flux1-dev-fp8.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/unet/flux1-dev-fp8.safetensors"
}

CHECKPOINT_URLS = {
    "": "",
    "animaPencilXLv500.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/checkpoints/animaPencilXLv500.safetensors",
    "atomixXL_v40.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/checkpoints/atomixXL_v40.safetensors",
    "copaxTimelessv12.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/checkpoints/copaxTimelessv12.safetensors",
    "dreamshaperXLv21.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/checkpoints/dreamshaperXLv21.safetensors",
    "juggernautXL_v8Rundiffusion.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/checkpoints/juggernautXL_v8Rundiffusion.safetensors",
    "leosamsHelloworldXL_helloworldXL70.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/checkpoints/leosamsHelloworldXL_helloworldXL70.safetensors",
    "Realistic5v5.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/checkpoints/Realistic5v5.safetensors",
    "samaritan3dCartoon_v40SDXL.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/checkpoints/samaritan3dCartoon_v40SDXL.safetensors",
    "sd3_medium.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/checkpoints/sd3_medium.safetensors",
    "sd3.5_medium.safetensors" : "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/checkpoints/sd3.5_medium.safetensors",
    "sd3.5_large.safetensors" : "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/checkpoints/sd3.5_large.safetensors",
    "sd3.5_large_fp8_scaled.safetensors" : "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/checkpoints/sd3.5_large_fp8_scaled.safetensors",
    "sd3.5_medium.safetensors" : "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/checkpoints/sd3.5_medium.safetensors",
    "sd3_medium.safetensors" : "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/checkpoints/sd3_medium.safetensors"
}

VAE_URLS = {
    "": "",
    "ae.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/vae/ae.safetensors",
    "sdxl_vae.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/vae/sdxl_vae.safetensors"
}

CLIP_URLS = {
    "": "",
    "clip_l.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/clip/clip_l.safetensors",
    "t5xxl_fp8_e4m3fn.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/clip/t5xxl_fp8_e4m3fn.safetensors",
    "t5xxl_fp16.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/clip/t5xxl_fp16.safetensors",
    "ViT-L-14.pt": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/clip/ViT-L-14.pt"
}

CLIP_VISION_URLS = {
    "": "",
    "clip_vision_g.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/clip_vision/clip_vision_g.safetensors",
    "clip_vision_vit_h.safetensors": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/clip_vision/clip_vision_vit_h.safetensors",
    "wd-v1-4-moat-tagger-v2.csv": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/clip_vision/wd-v1-4-moat-tagger-v2.csv"
}

UPSCALE_URLS = {
    "": "",
    "4xNMKDSuperscale.pt": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/upscale_models/4xNMKDSuperscale.pt", 
    "8xNMKDSuperscale.pt": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/upscale_models/8xNMKDSuperscale.pt",
    "fooocus_upscaler.bin": "https://monpetitcoindeweb.myqnapcloud.com:8081/diffusionia/upscale_models/fooocus_upscaler.bin"
}

# Liste des modèles Ollama disponibles
available_models = [
    "",
    "aya",
    "bakllava",
    "codegemma",
    "codellama",
    "codeqwen",
    "qwen2.5-coder",
    "qwen2.5-coder:7b",
    "deepseek-coder",
    "deepseek-coder-v2",
    "dolphin-llama3",
    "dolphin-mixtral",
    "dolphin-mistral",
    "gemma",
    "gemma2",
    "hermes3",
    "llama2",
    "llama2-uncensored",
    "llama3",
    "llama3.1",
    "llama3-chatqa",
    "llama3-gradient",
    "llava",
    "llava-llama3",
    "llava-phi3",
    "magicoder",
    "minicpm-v",
    "mistral",
    "mistral-large",
    "mistral-nemo",
    "mistral-openorca",
    "mixtral",
    "mxbai-embed-large",
    "nous-hermes",
    "nous-hermes2",
    "nomic-embed-text",
    "orca-mini",
    "phi",
    "phi3",
    "phi3.5",
    "phind-codellama",
    "qwen",
    "qwen2",
    "qwen2.5",
    "qwen2-math",
    "starcoder",
    "starcoder2",
    "starling-lm",
    "wizard-vicuna-uncensored",
    "wizardlm",
    "wizardlm2"
]

# Fonction pour récupérer les modèles Ollama installés
def get_installed_ollama_models():
    try:
        models = subprocess.check_output(["ollama", "list"], stderr=subprocess.DEVNULL).decode("utf-8")
        installed_models = [model.split(":")[0].strip() for model in models.splitlines() if model]
        return installed_models
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de 'ollama list': {e}")
        return []

# Fonction pour télécharger un modèle Ollama si non installé
import subprocess
import re

# Fonction pour nettoyer les séquences d'échappement ANSI
def clean_ansi_sequences(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

# Fonction pour filtrer les lignes avec une progression (par exemple, un pourcentage)
def is_progress_line(text):
    # On cherche un pourcentage suivi de chiffres (ex: "11%" ou "99 MB/4.7 GB") et on filtre les identifiants comme 'ea025c107c1c...'
    return re.search(r'\d+%.*\d+ [MG]B/\d+\.\d+ [MG]B', text)

# Fonction mise à jour pour télécharger un modèle et afficher la progression proprement
def pull_ollama_model(model_name):
    installed_models = get_installed_ollama_models()
    if model_name not in installed_models:
        try:
            yield f"Téléchargement du modèle '{model_name}':\n"  # Saut de ligne après le nom du modèle
            # Lancer le processus en capturant stdout et stderr ensemble
            process = subprocess.Popen(
                ["ollama", "pull", model_name], 
                stdout=subprocess.PIPE,  # Capturer stdout
                stderr=subprocess.STDOUT,  # Rediriger stderr vers stdout
                text=True,  # Travailler en mode texte
                encoding='utf-8'  # Utiliser UTF-8 pour décoder la sortie
            )
            
            # Lire la sortie en temps réel, ligne par ligne
            for line in iter(process.stdout.readline, ''):
                cleaned_line = clean_ansi_sequences(line.strip())  # Nettoyer la ligne
                if cleaned_line and is_progress_line(cleaned_line):  # Filtrer les lignes de progression
                    # Supprimer l'ID du modèle (comme 'ea025c107c1c...')
                    cleaned_line = re.sub(r'pulling [a-z0-9]+', ' pulling', cleaned_line)
                    yield f"{cleaned_line}"  # Retourner la ligne proprement formatée
            
            process.stdout.close()  # Fermer stdout après avoir lu toutes les lignes
            process.wait()  # Attendre la fin du processus

            if process.returncode == 0:
                yield f"Le modèle {model_name} a été téléchargé avec succès."
            else:
                yield f"Erreur lors du téléchargement de {model_name} avec ollama pull."
        
        except subprocess.CalledProcessError as e:
            yield f"Erreur lors du téléchargement de {model_name} : {e}"
    else:
        yield f"Le modèle {model_name} est déjà installé."

# Function to download large files with progress tracking in Gradio
def download_file(url, save_dir):
    local_filename = os.path.join(save_dir, url.split("/")[-1])
    if os.path.exists(local_filename):
        local_size = os.path.getsize(local_filename)
        try:
            response = requests.head(url, timeout=10)
            response.raise_for_status()
            remote_size = int(response.headers.get('content-length', 0))
            if local_size == remote_size:
                yield f"{local_filename} est déjà présent et a la même taille, téléchargement ignoré."
                return
        except requests.exceptions.RequestException as e:
            yield f"Impossible de vérifier la taille du fichier distant pour {local_filename}: {str(e)}"
            return

    try:
        with requests.get(url, stream=True, timeout=10) as response:
            response.raise_for_status()
            total_length = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = f"{local_filename.split(os.sep)[-1]}: {downloaded / total_length * 100:.2f}% téléchargé"
                        yield progress
        yield f"{local_filename} téléchargement terminé avec succès."
    except requests.exceptions.RequestException as e:
        yield f"Échec du téléchargement pour {local_filename}: {str(e)}"

# Gradio interface
def download_selected(unet, checkpoint, vae, clip, clip_vision, upscale, ollama):
    if unet:
        yield from download_file(UNET_URLS[unet], UNET_DIR)
    if checkpoint:
        yield from download_file(CHECKPOINT_URLS[checkpoint], CHECKPOINT_DIR)
    if vae:
        yield from download_file(VAE_URLS[vae], VAE_DIR)
    if clip:
        yield from download_file(CLIP_URLS[clip], CLIP_DIR)
    if clip_vision:
        yield from download_file(CLIP_VISION_URLS[clip_vision], CLIP_VISION_DIR)
    if upscale:
        yield from download_file(UPSCALE_URLS[upscale], UPSCALE_DIR)
    if ollama:
        yield from pull_ollama_model(ollama)  # Utiliser `yield from` pour rediriger la progression






#############################################################################################################################################################################
# Gradio interface
#############################################################################################################################################################################




def launch_interface():
    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column():
                download_output = gr.Textbox(label="📥 Progression des téléchargements", lines=4, interactive=False, elem_id="Progression")

                with gr.Accordion("🌐 Téléchargements des modèles de diffusion pour Comfy UI", open=False):
                    unet_dropdown = gr.Dropdown(list(UNET_URLS.keys()), label="📥 Télécharger des modèles Flux.1 dans le dossier unet", allow_custom_value=True)
                    checkpoint_dropdown = gr.Dropdown(list(CHECKPOINT_URLS.keys()), label="📥 Télécharger des modèles SDXL dans le dossier checkpoints", allow_custom_value=True)
                    vae_dropdown = gr.Dropdown(list(VAE_URLS.keys()), label="📥 Télécharger des modèles VAE dans le dossier vae", allow_custom_value=True)
                    clip_dropdown = gr.Dropdown(list(CLIP_URLS.keys()), label="📥 Télécharger des modèles Clip dans le dossier clip", allow_custom_value=True)
                    clip_vision_dropdown = gr.Dropdown(list(CLIP_VISION_URLS.keys()), label="📥 Télécharger des modèles Clip Vision dans le dossier clip_vision", allow_custom_value=True)
                    upscale_dropdown = gr.Dropdown(list(UPSCALE_URLS.keys()), label="📥 Télécharger des modèles Upscale dans le dossier upscale_models", allow_custom_value=True)

                with gr.Accordion("🌐 Téléchargements et installation des modèles Ollama", open=False):
                    ollama_dropdown = gr.Dropdown(available_models, label="💻 Liste des modèles IA Ollama 🦙", allow_custom_value=True)

                download_button = gr.Button("📥 Démarrer les Téléchargements")

            with gr.Column():
                gr.Markdown("""

                # Documentations
                - Gradio : [Documentations](https://gradio.app/docs/)
                - Ollama : [Documentations](https://github.com/ollama/ollama/tree/main/docs)
                - ComfyUI : [Documentations](https://github.com/comfyanonymous/ComfyUI?tab=readme-ov-file#readme)                  
                          
                ## Modèles LLM & Diffusion
                - Hugging Face : [Modèles](https://huggingface.co/models)
                - Ollama : [Modèles](https://ollama.ai/library)
                - Civitai : [Flux.1](https://civitai.com/search/models?modelType=LORA&modelType=Checkpoint&sortBy=models_v9&query=flux.1) [SDXL](https://civitai.com/search/models?modelType=Checkpoint&sortBy=models_v9&query=SDXL)
                            
                ## Liens Utiles
                - [GitHub de Gradio](https://github.com/gradio-app/gradio)
                - [Blog Hugging Face](https://huggingface.co/blog)
                - [Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
                            
                """)

        download_button.click(download_selected, 
                      [unet_dropdown, checkpoint_dropdown, vae_dropdown, clip_dropdown, clip_vision_dropdown, upscale_dropdown, ollama_dropdown], 
                      outputs=download_output, 
                      show_progress=True)  # Activer le suivi de progression


    return interface

launch_interface()
