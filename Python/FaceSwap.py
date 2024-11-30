import gradio as gr
import json
import requests
import os
import shutil
import tempfile
import time
from pathlib import Path

# Configuration des chemins
comfyUI_path = os.path.expandvars(r"%userprofile%\ComfyUI\ComfyUI")
OUTPUT_DIR = os.path.join(comfyUI_path, "output")

# Fonction pour effectuer le face swap et l'upscale en utilisant l'API
def process_images(image1, image2, swap_model, facedetection, face_restore_model, face_restore_visibility, codeformer_weight, detect_gender_input, detect_gender_source, input_faces_index, source_faces_index, console_log_level, upscale_method, scale_by):
    # Préparer les données pour l'API dans un format qui respecte la structure attendue
    api_data = {
        "prompt": {
            "270": {
                "inputs": {
                    "image": image1,
                    "upload": "image"
                },
                "class_type": "LoadImage",
                "_meta": {
                    "title": "input_1"
                }
            },
            "271": {
                "inputs": {
                    "image": image2,
                    "upload": "image"
                },
                "class_type": "LoadImage",
                "_meta": {
                    "title": "input_2"
                }
            },
            "257": {
                "inputs": {
                    "enabled": True,
                    "swap_model": swap_model,
                    "facedetection": facedetection,
                    "face_restore_model": face_restore_model,
                    "face_restore_visibility": face_restore_visibility,
                    "codeformer_weight": codeformer_weight,
                    "detect_gender_input": detect_gender_input,
                    "detect_gender_source": detect_gender_source,
                    "input_faces_index": input_faces_index,
                    "source_faces_index": source_faces_index,
                    "console_log_level": console_log_level,
                    "input_image": ["271", 0],
                    "source_image": ["270", 0]
                },
                "class_type": "ReActorFaceSwap",
                "_meta": {
                    "title": "reactor_faceswap"
                }
            },
            "325": {
                "inputs": {
                    "upscale_method": upscale_method,
                    "scale_by": scale_by,
                    "image": ["257", 0]
                },
                "class_type": "ImageScaleBy",
                "_meta": {
                    "title": "Upscale Image By"
                }
            },
            "318": {
                "inputs": {
                    "filename_prefix": "FaceSwap",
                    "images": ["325", 0]
                },
                "class_type": "SaveImage",
                "_meta": {
                    "title": "output"
                }
            }
        }
    }

    # Envoyer les données à l'API (remplacez par l'URL réelle de votre API)
    api_url = "http://127.0.0.1:8188/prompt"
    try:
        # Prendre un timestamp avant l'appel de l'API
        start_time = time.time()

        response = requests.post(api_url, json=api_data)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Attendre jusqu'à ce que l'image soit générée
        timeout = 60  # Timeout après 60 secondes
        latest_image_paths = []

        while time.time() - start_time < timeout:
            # Liste des fichiers dans le dossier de sortie
            output_files = get_latest_images_SDXL(OUTPUT_DIR, start_time)

            if output_files:
                latest_image_paths = output_files
                break
            
            # Attendre avant de vérifier à nouveau
            time.sleep(1)

        if not latest_image_paths:
            raise ValueError("Aucune image de sortie trouvée dans le dossier de sortie après le délai d'attente.")

        # Copier les images générées dans un répertoire temporaire
        temp_dir = tempfile.mkdtemp()
        temp_image_paths = []
        for img_path in latest_image_paths:
            temp_image_path = shutil.copy(img_path, temp_dir)
            temp_image_paths.append(temp_image_path)

    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {e}")
    except ValueError as e:
        raise Exception(f"API response error: {e}")

    # Retourner le chemin des fichiers temporaires pour les afficher dans la galerie
    return temp_image_paths

# Fonction pour obtenir les dernières images générées après un timestamp donné
def get_latest_images_SDXL(folder, start_time):
    files = os.listdir(folder)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    new_images = [os.path.join(folder, f) for f in image_files if os.path.getmtime(os.path.join(folder, f)) > start_time]
    return new_images

# Interface Gradio
import gradio as gr

def launch_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                image1 = gr.Image(label="Image 1", type="filepath", height=270)
                image2 = gr.Image(label="Image 2", type="filepath", height=270)

                with gr.Accordion("Parameters", open=False):
                    with gr.Tab("Reactor Face Swap"):
                        swap_model = gr.Dropdown(label="Swap Model", choices=["inswapper_128.onnx"], value="inswapper_128.onnx")
                        facedetection = gr.Dropdown(label="Face Detection", choices=["retinaface_resnet50"], value="retinaface_resnet50")
                        face_restore_model = gr.Dropdown(label="Face Restore Model", choices=["GPEN-BFR-1024.onnx"], value="GPEN-BFR-1024.onnx")
                        face_restore_visibility = gr.Slider(label="Face Restore Visibility", minimum=0, maximum=1, step=0.01, value=1.0)
                        codeformer_weight = gr.Slider(label="Codeformer Weight", minimum=0, maximum=1, step=0.01, value=1.0)
                        detect_gender_input = gr.Dropdown(label="Detect Gender Input", choices=["no"], value="no")
                        detect_gender_source = gr.Dropdown(label="Detect Gender Source", choices=["no"], value="no")
                        input_faces_index = gr.Number(label="Input Faces Index", value=0)
                        source_faces_index = gr.Number(label="Source Faces Index", value=0)
                        console_log_level = gr.Number(label="Console Log Level", value=1)

                    with gr.Tab("Upscale Parameters"):
                        upscale_method = gr.Dropdown(label="Upscale Method", choices=["nearest-exact"], value="nearest-exact")
                        scale_by = gr.Number(label="Scale By", value=4)

            with gr.Column(scale=2):
                output_gallery = gr.Gallery(
                    label="Galerie d'images",
                    height=550,
                    show_label=True,
                    allow_preview=True,
                    preview=True,
                    object_fit="contain",
                    show_download_button=True
                )
                run_button = gr.Button("Run")

        run_button.click(
            process_images,
            inputs=[
                image1, image2, swap_model, facedetection, face_restore_model, face_restore_visibility,
                codeformer_weight, detect_gender_input, detect_gender_source, input_faces_index, source_faces_index,
                console_log_level, upscale_method, scale_by
            ],
            outputs=output_gallery
        )

    return demo

if __name__ == "__main__":
    # Ajouter le chemin de sortie comme chemin autorisé
    launch_interface().launch(allowed_paths=[OUTPUT_DIR])
