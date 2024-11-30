import json
import os
import time
import gradio as gr
from pathlib import Path
import requests
import shutil
import tempfile

# Directories and API URL
URL = "http://127.0.0.1:8188/prompt"
base_path = os.path.expandvars(r"%userprofile%\eXtraLlama")
comfyUI_path = os.path.expandvars(r"%userprofile%\ComfyUI\ComfyUI")
INPUT_DIR = os.path.join(comfyUI_path, "input")
OUTPUT_DIR = os.path.join(comfyUI_path, "output")
UNET_DIRS = [os.path.join(comfyUI_path, "models", "unet")]
VAE_DIRS = [os.path.join(comfyUI_path, "models", "vae")]
CLIP_DIRS = [os.path.join(comfyUI_path, "models", "clip")]

# API workflow configuration
workflow_data = {
  "5": {"inputs": {"width": 1024, "height": 1024, "batch_size": 1}, "class_type": "EmptyLatentImage", "_meta": {"title": "Empty Latent Image"}},
  "6": {"inputs": {"text": "", "clip": ["11", 0]}, "class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text Encode (Prompt)"}},
  "8": {"inputs": {"samples": ["13", 0], "vae": ["10", 0]}, "class_type": "VAEDecode", "_meta": {"title": "VAE Decode"}},
  "9": {"inputs": {"filename_prefix": "ComfyUI_Flux-Dev", "images": ["8", 0]}, "class_type": "SaveImage", "_meta": {"title": "Save Image"}},
  "10": {"inputs": {"vae_name": "ae.safetensors"}, "class_type": "VAELoader", "_meta": {"title": "Load VAE"}},
  "11": {"inputs": {"clip_name1": "t5xxl_fp8_e4m3fn.safetensors", "clip_name2": "clip_l.safetensors", "type": "flux"}, "class_type": "DualCLIPLoader", "_meta": {"title": "DualCLIPLoader"}},
  "12": {"inputs": {"unet_name": "flux1-dev.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader", "_meta": {"title": "Load Diffusion Model"}},
  "13": {"inputs": {"noise": ["25", 0], "guider": ["22", 0], "sampler": ["16", 0], "sigmas": ["17", 0], "latent_image": ["5", 0]}, "class_type": "SamplerCustomAdvanced", "_meta": {"title": "SamplerCustomAdvanced"}},
  "16": {"inputs": {"sampler_name": "euler"}, "class_type": "KSamplerSelect", "_meta": {"title": "KSamplerSelect"}},
  "17": {"inputs": {"scheduler": "simple", "steps": 20, "denoise": 1, "model": ["12", 0]}, "class_type": "BasicScheduler", "_meta": {"title": "BasicScheduler"}},
  "22": {"inputs": {"model": ["12", 0], "conditioning": ["6", 0]}, "class_type": "BasicGuider", "_meta": {"title": "BasicGuider"}},
  "25": {"inputs": {"noise_seed": 123456789}, "class_type": "RandomNoise", "_meta": {"title": "RandomNoise"}}
}

def get_new_images(folder, start_time):
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    new_images = [os.path.join(folder, f) for f in image_files if os.path.getmtime(os.path.join(folder, f)) > start_time]
    new_images.sort(key=lambda x: os.path.getmtime(x))
    return new_images

def start_queue(prompt_workflow):
    data = json.dumps({"prompt": prompt_workflow}).encode('utf-8')
    requests.post(URL, data=data)

def list_files(directories, extensions):
    files = []
    for directory in directories:
        for file in Path(directory).glob("*"):
            if file.suffix in extensions:
                files.append(file.name)
    return files

def generate_image(clip_text, unet_name, clip_name1, clip_name2, vae_name, width, height, batch_size, noise_seed, sampler_name, steps):
    prompt = workflow_data

    prompt["6"]["inputs"]["text"] = clip_text
    prompt["10"]["inputs"]["vae_name"] = vae_name
    prompt["11"]["inputs"]["clip_name1"] = clip_name1
    prompt["11"]["inputs"]["clip_name2"] = clip_name2
    prompt["12"]["inputs"]["unet_name"] = unet_name
    prompt["5"]["inputs"]["width"] = width
    prompt["5"]["inputs"]["height"] = height
    prompt["5"]["inputs"]["batch_size"] = batch_size
    prompt["25"]["inputs"]["noise_seed"] = noise_seed
    prompt["16"]["inputs"]["sampler_name"] = sampler_name
    prompt["17"]["inputs"]["steps"] = steps

    start_time = time.time()
    start_queue(prompt)

    timeout = 600
    while True:
        new_images = get_new_images(OUTPUT_DIR, start_time)
        if len(new_images) >= batch_size:
            # Move images to a temporary directory to comply with Gradio's restrictions
            temp_dir = tempfile.gettempdir()
            temp_images = []
            for img_path in new_images[:batch_size]:
                temp_path = os.path.join(temp_dir, os.path.basename(img_path))
                shutil.copy(img_path, temp_path)
                temp_images.append(temp_path)
            return temp_images
        if time.time() - start_time > timeout:
            return ["Erreur : le délai d'attente a été dépassé pendant la génération de l'image."]
        time.sleep(1)

def launch_interface():
    unet_files = list_files(UNET_DIRS, [".safetensors", ".ckpt", ".gguf"])
    vae_files = list_files(VAE_DIRS, [".safetensors", ".ckpt", ".gguf"])
    clip_files = list_files(CLIP_DIRS, [".safetensors", ".ckpt", ".gguf"])

    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column(scale=1):
                clip_text = gr.Textbox(
                    lines=4,
                    label="🎚️ Prompt général", 
                    value="stairway to heaven and the clouds in the blue sky gather to form the words: welcome Flux", 
                    info="Visuels concis, attributs stylistiques.",
                    placeholder="Écrire le prompt ici"
                )
                unet_name = gr.Dropdown(
                    choices=unet_files,
                    value="flux1-dev.safetensors" if unet_files else None,
                    label="🎚️ Choix du checkpoint",
                    allow_custom_value=True
                )
                with gr.Accordion(" 🎛️ Paramètres", open=False):
                    clip_name1 = gr.Dropdown(
                        choices=clip_files, 
                        value="clip_l.safetensors", 
                        allow_custom_value=True, 
                        label="🎚️ CLIP 1"
                    )
                    clip_name2 = gr.Dropdown(
                        choices=clip_files, 
                        value="t5xxl_fp8_e4m3fn.safetensors", 
                        allow_custom_value=True, 
                        label="🎚️ CLIP 2"
                    )
                    vae_name = gr.Dropdown(
                        choices=vae_files, 
                        value="ae.safetensors", 
                        allow_custom_value=True, 
                        label="🎚️ VAE"
                    )
                    noise_seed = gr.Number(
                        value=123456789, 
                        label="🎚️ Seed"
                    )
                    sampler_name = gr.Dropdown(
                        choices=["euler", "ddim", "plms"], 
                        value="euler", 
                        label="🎚️ Sampler"
                    )
                    steps = gr.Slider(
                        minimum=1, maximum=100, value=20, step=1, label="🎚️ Steps"
                    )
                width = gr.Slider(
                    minimum=512, maximum=2048, value=1280, step=256, label="🎚️ Largeur de l'image"
                )
                height = gr.Slider(
                    minimum=512, maximum=2048, value=768, step=256, label="🎚️ Hauteur de l'image"
                )
                batch_size = gr.Slider(
                    minimum=1, maximum=12, value=4, step=1, label="🎚️ Nombre d'images"
                )

            with gr.Column(scale=2):
                output_gallery = gr.Gallery(
                    label="Galerie d'images",
                    height=505,
                    show_label=True,
                    allow_preview=True,
                    object_fit="contain",
                    show_download_button=True,
                    elem_id="custom-gallery"
                )
                generate_button = gr.Button("Générer les images")
                generate_button.click(
                    fn=generate_image,
                    inputs=[clip_text, unet_name, clip_name1, clip_name2, vae_name, width, height, batch_size, noise_seed, sampler_name, steps],
                    outputs=output_gallery
                )
    return interface

if __name__ == "__main__":
    launch_interface().launch(allowed_paths=[OUTPUT_DIR])
