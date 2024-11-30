import os
import json
import time
import gradio as gr
import requests
import shutil
import tempfile
from pathlib import Path

# Directories for models and outputs
URL = "http://127.0.0.1:8188/prompt"
base_path = os.path.expandvars(r"%userprofile%\eXtraLlama")
comfyUI_path = os.path.expandvars(r"%userprofile%\ComfyUI\ComfyUI")
INPUT_DIR = os.path.join(comfyUI_path, "input")
OUTPUT_DIR = os.path.join(comfyUI_path, "output")
CHECKPOINT_DIRS = [os.path.join(comfyUI_path, "models", "checkpoints")]
LORA_DIRS = [os.path.join(comfyUI_path, "models", "loras")]
VAE_DIRS = [os.path.join(comfyUI_path, "models", "vae")]
CLIP_DIRS = [os.path.join(comfyUI_path, "models", "clip")]

# Workflow JSON template
workflow_template = {
    "6": {
        "inputs": {
            "text": "",
            "clip": ["11", 0]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
            "title": "CLIP Text Encode (Prompt)"
        }
    },
    "11": {
        "inputs": {
            "clip_name1": "clip_g.safetensors",
            "clip_name2": "clip_l.safetensors",
            "clip_name3": "t5xxl_fp16.safetensors"
        },
        "class_type": "TripleCLIPLoader",
        "_meta": {
            "title": "TripleCLIPLoader"
        }
    },
    "13": {
        "inputs": {
            "shift": 3,
            "model": ["252", 0]
        },
        "class_type": "ModelSamplingSD3",
        "_meta": {
            "title": "ModelSamplingSD3"
        }
    },
    "67": {
        "inputs": {
            "conditioning": ["71", 0]
        },
        "class_type": "ConditioningZeroOut",
        "_meta": {
            "title": "ConditioningZeroOut"
        }
    },
    "68": {
        "inputs": {
            "start": 0.1,
            "end": 1,
            "conditioning": ["67", 0]
        },
        "class_type": "ConditioningSetTimestepRange",
        "_meta": {
            "title": "ConditioningSetTimestepRange"
        }
    },
    "69": {
        "inputs": {
            "conditioning_1": ["68", 0],
            "conditioning_2": ["70", 0]
        },
        "class_type": "ConditioningCombine",
        "_meta": {
            "title": "Conditioning (Combine)"
        }
    },
    "70": {
        "inputs": {
            "start": 0,
            "end": 0.1,
            "conditioning": ["71", 0]
        },
        "class_type": "ConditioningSetTimestepRange",
        "_meta": {
            "title": "ConditioningSetTimestepRange"
        }
    },
    "71": {
        "inputs": {
            "text": "bad quality, poor quality, doll, disfigured, jpg, toy, bad anatomy, missing limbs, missing fingers, 3d, cgi, text,",
            "clip": ["11", 0]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
            "title": "CLIP Text Encode (Negative Prompt)"
        }
    },
    "135": {
        "inputs": {
            "width": 1024,
            "height": 1024,
            "batch_size": 1
        },
        "class_type": "EmptySD3LatentImage",
        "_meta": {
            "title": "EmptySD3LatentImage"
        }
    },
    "231": {
        "inputs": {
            "samples": ["271", 0],
            "vae": ["252", 2]
        },
        "class_type": "VAEDecode",
        "_meta": {
            "title": "VAE Decode"
        }
    },
    "252": {
        "inputs": {
            "ckpt_name": "sd3_medium.safetensors"
        },
        "class_type": "CheckpointLoaderSimple",
        "_meta": {
            "title": "Load Checkpoint"
        }
    },
    "271": {
        "inputs": {
            "seed": 596905809080229,
            "steps": 28,
            "cfg": 4.5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "sgm_uniform",
            "denoise": 1,
            "model": ["13", 0],
            "positive": ["6", 0],
            "negative": ["69", 0],
            "latent_image": ["135", 0]
        },
        "class_type": "KSampler",
        "_meta": {
            "title": "KSampler"
        }
    },
    "273": {
        "inputs": {
            "filename_prefix": "ComfyUI_SD3",
            "images": ["231", 0]
        },
        "class_type": "SaveImage",
        "_meta": {
            "title": "Save Image"
        }
    }
}

# Manual models list
manual_models = [
    "sd3.5_large.safetensors",
    "sd3.5_large_fp8_scaled.safetensors",
    "sd3.5_medium.safetensors",
    "sd3_medium.safetensors",
    "clip_g.safetensors",
    "clip_l.safetensors",
    "t5xxl_fp16.safetensors"
]

# Function to list files from directories
def list_files(directories, extensions):
    checkpoint_files = []
    for directory in directories:
        for file in Path(directory).glob("*"):
            if file.suffix in extensions and file.name in manual_models:
                checkpoint_files.append(file.stem)  # Remove extension
    return checkpoint_files

# Function to send a prompt to the new API
def start_queue(prompt_workflow):
    data = json.dumps({"prompt": prompt_workflow}).encode('utf-8')
    requests.post(URL, data=data)

# Function to generate images using the new API
def generate_image(checkpoint_file, text_prompt, negative_prompt, clip1, clip2, clip3, width, height, batch_size, steps, cfg, sampler_name, seed):
    # Update the workflow fields with user inputs
    prompt = workflow_template.copy()
    prompt["252"]["inputs"]["ckpt_name"] = checkpoint_file + ".safetensors"
    prompt["6"]["inputs"]["text"] = text_prompt
    prompt["71"]["inputs"]["text"] = negative_prompt
    prompt["11"]["inputs"]["clip_name1"] = clip1 + ".safetensors"
    prompt["11"]["inputs"]["clip_name2"] = clip2 + ".safetensors"
    prompt["11"]["inputs"]["clip_name3"] = clip3 + ".safetensors"
    prompt["135"]["inputs"]["width"] = width
    prompt["135"]["inputs"]["height"] = height
    prompt["135"]["inputs"]["batch_size"] = batch_size
    prompt["271"]["inputs"]["steps"] = steps
    prompt["271"]["inputs"]["cfg"] = cfg
    prompt["271"]["inputs"]["sampler_name"] = sampler_name
    prompt["271"]["inputs"]["seed"] = seed

    # Start the queue in the new API
    start_queue(prompt)

    # Implement a waiting mechanism to retrieve the generated images
    start_time = time.time()
    timeout = 600  # Maximum wait time in seconds
    generated_images = []
    while True:
        new_images = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.getmtime(os.path.join(OUTPUT_DIR, f)) > start_time]
        if len(new_images) >= batch_size:
            generated_images = [os.path.join(OUTPUT_DIR, img) for img in new_images[:batch_size]]
            break

        if time.time() - start_time > timeout:
            return ["Error: Timeout exceeded during image generation."]

        time.sleep(1)

    # Move images to a temporary directory to comply with Gradio's restrictions
    temp_dir = tempfile.gettempdir()
    temp_images = []
    for img_path in generated_images:
        temp_path = os.path.join(temp_dir, os.path.basename(img_path))
        shutil.copy(img_path, temp_path)
        temp_images.append(temp_path)

    return temp_images

# Gradio Interface
def launch_interface():
    clip_files = list_files(CLIP_DIRS, [".safetensors", ".ckpt"])
    checkpoint_files = list_files(CHECKPOINT_DIRS, [".safetensors", ".ckpt"])
    
    with gr.Blocks() as interface:      
        with gr.Row():
            with gr.Column():
                text_prompt = gr.Textbox(
                    lines=3,
                    label="🎚️ Prompt positif",
                    placeholder="Écrire le prompt positif ici",
                    value="Vue frontale en gros plan d'une femme robot cyborg séduisante et sexy portant des lunettes VR avec les mots de couleur verts lumineux \"Expo 24\" écrits clairement sur les lunettes VR, sur fond de datacenter futuriste aux couleurs saturées.",
                    info="Ce qui doit s'afficher",
                    elem_id="text_prompt"
                )
                negative_prompt = gr.Textbox(
                    lines=3,
                    label="🎚️ Prompt négatif",
                    placeholder="Écrire le prompt négatif ici",
                    value="mauvaise qualité, mains mal dessinées, visage mal dessiné, doigts supplémentaires ou manquants, mains et visage déformés, proportions corporelles irréalistes, membres surnuméraires, doigts fusionnés, flou, mauvaise anatomie, textes, watermarks, mauvaise symétrie, logos.",
                    info="Ce qui ne doit pas s'afficher",
                    elem_id="negative_prompt"
                )
                checkpoint_file = gr.Dropdown(
                    choices=checkpoint_files,
                    value="sd3_medium" if checkpoint_files else None,
                    label="🎚️ Choix du checkpoint"
                )   
                with gr.Accordion(" 🎛️ Paramètres", open=False):
                    with gr.Column(scale=1):
                        steps = gr.Slider(
                            minimum=1, 
                            maximum=100, 
                            value=20, 
                            step=1, 
                            label="🎚️ Steps"
                        )
                        cfg = gr.Slider(
                            minimum=0.1, 
                            maximum=10, 
                            value=4.5, 
                            step=0.1, 
                            label="🎚️ CFG"
                        )
                        sampler_name = gr.Dropdown(
                            label="🎚️ Sampler", 
                            choices=["dpmpp_2m","euler", "heun"], 
                            value="euler"    
                        )
                        seed = gr.Number(
                            value=123456789,
                            label="🎚️ Seed"
                        )
                        clip1 = gr.Dropdown(
                            choices=clip_files,
                            value="clip_g" if clip_files else None,
                            label="🎚️ CLIP Model 1"
                        )
                        clip2 = gr.Dropdown(
                            choices=clip_files,
                            value="clip_l" if clip_files else None,
                            label="🎚️ CLIP Model 2"
                        )
                        clip3 = gr.Dropdown(
                            choices=clip_files,
                            value="t5xxl_fp16" if clip_files else None,
                            label="🎚️ CLIP Model 3"
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
                    output = gr.Gallery(
                        label="Galerie d'images",
                        height=640,
                        show_label=True,
                        allow_preview=True,
                        preview=True,
                        object_fit="contain",
                        show_download_button=True,
                        elem_id="custom-gallery"
                    )
                    generate_button = gr.Button("Générer les images")

                    generate_button.click(
                        fn=generate_image,
                        inputs=[checkpoint_file, text_prompt, negative_prompt, clip1, clip2, clip3, width, height, batch_size, steps, cfg, sampler_name, seed],
                        outputs=output
                    )

    return interface

if __name__ == "__main__":
    launch_interface().launch(allowed_paths=[OUTPUT_DIR])
