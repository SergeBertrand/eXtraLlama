# eXtraLlama

## Description

Portail UI Gradio pour Ollama & ComfyUI

## Prérequis

- Windows 10/11 x64
- Python 3.10
- CUDA 11.8.0
- Visual Studio
- ComfyUI Portable (`%profileuser%\ComfyUI\ComfyUI`)
- Ollama

### Modèles Ollama

- Llama3.1
- Llama3.2
- Llama3.2-vision
- Mistral

### Modèles & Custom Nodes pour ComfyUI

#### Dans le dossier : `ComfyUI\ComfyUI\models\checkpoints`

- animaPencilXLv500.safetensors
- atomixXL_v40.safetensors
- copaxTimelessv12.safetensors
- dreamshaperXLv21.safetensors
- juggernautXL_v8Rundiffusion.safetensors
- leosamsHelloworldXL_helloworldXL70.safetensors
- Realistic5v5.safetensors
- samaritan3dCartoon_v40SDXL.safetensors
- sd3.5_large.safetensors
- sd3.5_large_fp8_scaled.safetensors
- sd3.5_medium.safetensors
- sd3_medium.safetensors

#### Dans le dossier : `ComfyUI\ComfyUI\models\clip`

- clip_g.safetensors
- clip_l.safetensors
- t5xxl_fp8_e4m3fn.safetensors
- t5xxl_fp16.safetensors
- ViT-L-14.pt

#### Dans le dossier : `ComfyUI\ComfyUI\clip_vision`

- clip_vision_g.safetensors
- clip_vision_vit_h.safetensors

#### Dans le dossier : `ComfyUI\ComfyUI\unet`

- flux1-dev.safetensors
- flux1-schnell.safetensors
- flux1-dev-fp8.safetensors

#### Dans le dossier : `ComfyUI\ComfyUI\Custom_nodes`

- ComfyUI-Manager
- comfyui-reactor-node

## Installation

1. **Télécharger le projet** :
   - Renommer et copier à la racine du dossier utilisateur : `%userfile%\eXtraLlama`

2. **Utiliser Git pour cloner le projet** :
   - Cloner le projet à la racine du dossier utilisateur :
     ```sh
     git clone https://github.com/SergeBertrand/eXtraLlama.git %userfile%\eXtraLlama
     ```

## Workflows ComfyUI

### Exemple de Workflow SDXL

![SDXL_Realistic5v5](Fichiers/Workflows/SDXL_Realistic5v5.png)

![SDXL_samaritan3dCartoon_v40](Fichiers/Workflows/SDXL_samaritan3dCartoon_v40.png)

### Exemple de Workflow SD3 Medium

![SD3_Medium](Fichiers/Workflows/SD3_Medium.png)

### Exemple de Workflow SD3.5 Large

![SD3.5_Large](Fichiers/Workflows/SD3.5_Large.png)

### Exemple de Workflow 2

![Flux-Dev](Fichiers/Workflows/Flux-Dev.png)

## Contributions

Nous encourageons les contributions de la communauté pour améliorer ce projet. Si vous souhaitez contribuer, veuillez suivre ces étapes :

1. **Forker le dépôt** : Créez une copie de ce dépôt sur votre propre compte GitHub.
2. **Créer une branche** : Créez une nouvelle branche pour votre contribution.
3. **Faire des modifications** : Apportez vos modifications et améliorations.
4. **Soumettre une Pull Request** : Soumettez une Pull Request pour que vos modifications soient revues et fusionnées.

## Licence

Ce projet est distribué sous la licence MIT. Vous êtes libre d'utiliser, de modifier et de distribuer ce code à condition de conserver la mention de la licence originale. Pour plus de détails, consultez le fichier `LICENSE`.
