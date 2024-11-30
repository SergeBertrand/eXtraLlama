import os
import json
import time
import requests
import gradio as gr
from pathlib import Path
import shutil
import tempfile

URL = "http://127.0.0.1:8188/prompt"
comfyUI_path = os.path.expandvars(r"%userprofile%\ComfyUI\ComfyUI")
INPUT_DIR = os.path.join(comfyUI_path, "input")
OUTPUT_DIR = os.path.join(comfyUI_path, "output")
CHECKPOINT_DIRS = [os.path.join(comfyUI_path, "models", "checkpoints")]

workflow_data = {
    "3": {
        "inputs": {
            "seed": 123456789,
            "steps": 20,
            "cfg": 8,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0]
        },
        "class_type": "KSampler",
        "_meta": {
            "title": "KSampler"
        }
    },
    "4": {
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.ckpt"
        },
        "class_type": "CheckpointLoaderSimple",
        "_meta": {
            "title": "Load Checkpoint"
        }
    },
    "5": {
        "inputs": {
            "width": 512,
            "height": 512,
            "batch_size": 1
        },
        "class_type": "EmptyLatentImage",
        "_meta": {
            "title": "Empty Latent Image"
        }
    },
    "6": {
        "inputs": {
            "text": "",
            "clip": ["4", 1]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
            "title": "CLIP Text Encode (Prompt)"
        }
    },
    "7": {
        "inputs": {
            "text": "",
            "clip": ["4", 1]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
            "title": "CLIP Text Encode (Prompt)"
        }
    },
    "8": {
        "inputs": {
            "samples": ["3", 0],
            "vae": ["4", 2]
        },
        "class_type": "VAEDecode",
        "_meta": {
            "title": "VAE Decode"
        }
    },
    "9": {
        "inputs": {
            "filename_prefix": "ComfyUI_SDXL",
            "images": ["8", 0]
        },
        "class_type": "SaveImage",
        "_meta": {
            "title": "Save Image"
        }
    }
}

def get_latest_images_SDXL(folder):
    files = os.listdir(folder)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
    return [os.path.join(folder, f) for f in image_files]

def start_queue_SDXL(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    requests.post(URL, data=data)

def list_manual_checkpoint_files():
    manual_models = [
        "animaPencilXLv500.safetensors",
        "atomixXL_v40.safetensors",
        "copaxTimelessv12.safetensors",
        "dreamshaperXLv21.safetensors",
        "juggernautXL_v8Rundiffusion.safetensors",
        "leosamsHelloworldXL_helloworldXL70",
        "Realistic5v5.safetensors",
        "samaritan3dCartoon_v40SDXL.safetensors"
    ]

    available_models = []
    for model in manual_models:
        for directory in CHECKPOINT_DIRS:
            model_path = os.path.join(directory, model)
            if os.path.isfile(model_path):
                available_models.append(model.replace(".safetensors", ""))  # Masquer l'extension
    return available_models

def generate_image_SDXL(positive_prompt_text, negative_prompt_text, checkpoint_file, output_width, output_height, batch_size, sampler_name, cfg, steps, seed):
    prompt = workflow_data.copy()
    prompt["6"]["inputs"]["text"] = f"digital artwork of a {positive_prompt_text}"
    prompt["7"]["inputs"]["text"] = f"{negative_prompt_text}"
    prompt["4"]["inputs"]["ckpt_name"] = f"{checkpoint_file}.safetensors"  # Ajouter l'extension si nécessaire
    prompt["5"]["inputs"]["width"] = output_width
    prompt["5"]["inputs"]["height"] = output_height
    prompt["5"]["inputs"]["batch_size"] = batch_size
    prompt["3"]["inputs"]["sampler_name"] = sampler_name
    prompt["3"]["inputs"]["cfg"] = cfg
    prompt["3"]["inputs"]["steps"] = steps
    prompt["3"]["inputs"]["seed"] = seed

    start_time = time.time()
    start_queue_SDXL(prompt)

    timeout = 180
    while True:
        latest_images = get_latest_images_SDXL(OUTPUT_DIR)
        new_images = [img for img in latest_images if os.path.getmtime(img) > start_time]
        if len(new_images) >= batch_size:
            temp_dir = tempfile.mkdtemp()
            copied_images = []
            for img in new_images[:batch_size]:
                new_path = shutil.copy(img, temp_dir)
                copied_images.append(new_path)
            return copied_images

        if time.time() - start_time > timeout:
            return ["Erreur : le délai d'attente a été dépassé pendant la génération de l'image."]

        time.sleep(1)

def launch_interface():
    checkpoint_files = list_manual_checkpoint_files()
    
    with gr.Blocks(theme=gr.themes.Soft(spacing_size="sm", text_size="lg")) as interface:
        with gr.Row():
            with gr.Column(scale=1):
                positive_prompt_text = gr.Textbox(
                    lines=3,
                    label="🎚️ Prompt positif",
                    info="Ce qui doit s'afficher",
                    placeholder="Écrire le prompt positif ici",
                    value="Maître Yoda, en fond les marécages de la planète Dagobah."
                )
                negative_prompt_text = gr.Textbox(
                    lines=3,
                    label="🎚️ Prompt négatif",
                    info="Ce qui ne doit pas s'afficher",
                    placeholder="Écrire le prompt négatif ici",
                    value="mauvaise qualité, mains mal dessinées, visage mal dessiné, doigts supplémentaires ou manquants, mains et visage déformés, proportions corporelles irréalistes, membres surnuméraires, doigts fusionnés, flou, mauvaise anatomie, textes, watermarks, mauvaise symétrie, logos."
                )

                checkpoint_file = gr.Dropdown(
                    choices=checkpoint_files,
                    label="🎚️ Choix du checkpoint",
                    value="dreamshaperXLv21" if "dreamshaperXLv21" in checkpoint_files else (checkpoint_files[0] if checkpoint_files else None),
                    elem_id="checkpoint_file"
                )

                with gr.Accordion(" 🎛️ Paramètres", open=False):
                    sampler_name = gr.Dropdown(
                        label="🎚️ Sampler", 
                        choices=["dpmpp_2m", "euler", "heun"], 
                        value="euler"
                    )
                    cfg = gr.Slider(
                        label="🎚️ CFG", 
                        minimum=1, 
                        maximum=20, 
                        value=8
                    )
                    steps = gr.Slider(
                        label="🎚️ Steps", 
                        minimum=1, 
                        maximum=100, 
                        value=20
                    )
                    seed = gr.Slider(
                        label="🎚️ Seed", 
                        minimum=0, 
                        maximum=1e18, 
                        step=1, 
                        value=123456789
                    )

                output_width = gr.Slider(
                    minimum=512,
                    maximum=2048,
                    step=256,
                    label="🎚️ Largeur de l'image",
                    value=1280
                )
                output_height = gr.Slider(
                    minimum=512,
                    maximum=2048,
                    step=256,
                    label="🎚️ Hauteur de l'image",
                    value=768
                )
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=12,
                    step=1,
                    label="🎚️ Nombre d'images",
                    value=4
                )
                
            with gr.Column(scale=2):
                output_gallery = gr.Gallery(
                    label="Galerie d'images",
                    height=640,
                    show_label=True,
                    allow_preview=True,
                    preview=True,
                    object_fit="contain",
                    show_download_button=True
                )

                generate_button = gr.Button("Générer les images")
                generate_button.click(
                    fn=generate_image_SDXL,
                    inputs=[
                        positive_prompt_text,
                        negative_prompt_text,
                        checkpoint_file,
                        output_width,
                        output_height,
                        batch_size,
                        sampler_name,
                        cfg,
                        steps,
                        seed
                    ],
                    outputs=output_gallery
                )
                
    return interface

if __name__ == "__main__":
    launch_interface().launch(allowed_paths=[OUTPUT_DIR])
