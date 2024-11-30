import gradio as gr
import importlib.util
import os
import sys
import json
import time
import requests
from pathlib import Path
from pydantic import BaseModel
from starlette.requests import Request
import colorama
from colorama import Fore, Style
import logging
import urllib3
import warnings

# Ignorer les warnings
warnings.filterwarnings("ignore", category=FutureWarning)
colorama.init(autoreset=True)  # Initialiser colorama pour la gestion des couleurs

# CSS pour la grille de fonctionnalités
custom_css = """
/*----------------------------------------------------- Style pour personnaliser le Chatbot  -------------------------------------------------------------------*/


#custom-gallery .grid-container {
    height: 100vh !important; /* Ajustez la hauteur selon vos besoins */
    overflow: auto !important; /* Permet de défiler si le contenu dépasse la hauteur */
}

.selected {
    font-weight: bold !important;
    font-size: 18px !important;
    color: #4f46e5 !important;
}

/*--------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/* Styles pour la grille */
.feature-container {
    display: grid;
    grid-template-columns: repeat(3, 1fr); /* Trois colonnes */
    gap: 10px;
    margin-right: 10px;
    margin-left: 10px;
}

.feature-block {
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
#    border-radius: 8px;
    border-bottom: 3px solid #6569ec;
    margin: 0 20px;
    padding: 10px;
    text-align: left;
    min-height: 100px;
    box-sizing: border-box;
}

"""
# Mettre à jour le HTML des blocs pour les styles demandés
feature_blocks_html = """
<div class="feature-container">
    <div class="feature-block" style="background-color: Transparent;">
        <div class="emoji-title" style="display: flex; align-items: center;">
            <div class="feature-emoji" style="font-size: 35px; margin-right: 5px;">💬</div>
            <h3 style="color: #4f46e5; font-weight: bolder; margin: 0; font-size: 1.5em;">Chatbot</h3>
        </div>
        <p style="color: #9ca3af; font-weight: bold;">Un assistant virtuel IA qui vous permet de dialoguer avec des modèles LLM dans différents langages.</p>
    </div>

    <div class="feature-block" style="background-color: Transparent;">
        <div class="emoji-title" style="display: flex; align-items: center;">
            <div class="feature-emoji" style="font-size: 35px; margin-right: 5px;">📄</div>
            <h3 style="color: #4f46e5; font-weight: bolder; margin: 0; font-size: 1.5em;">PDF</h3>
        </div>
        <p style="color: #9ca3af; font-weight: bold;">Un assistant virtuel IA capable d'extraire et d'analyser des informations à partir de documents PDF.</p>
    </div>

    <div class="feature-block" style="background-color: Transparent;">
        <div class="emoji-title" style="display: flex; align-items: center;">
            <div class="feature-emoji" style="font-size: 35px; margin-right: 5px;">🎙️</div>
            <h3 style="color: #4f46e5; font-weight: bolder; margin: 0; font-size: 1.5em;">Transcript</h3>
        </div>
        <p style="color: #9ca3af; font-weight: bold;">Un assistant IA, capable de comprendre des questions verbales et de répondre en langage vocal.</p>
    </div>

    <div class="feature-block" style="background-color: Transparent;">
        <div class="emoji-title" style="display: flex; align-items: center;">
            <div class="feature-emoji" style="font-size: 35px; margin-right: 5px;">📢</div>
            <h3 style="color: #4f46e5; font-weight: bolder; margin: 0; font-size: 1.5em;">Commandes</h3>
        </div>
        <p style="color: #9ca3af; font-weight: bold;">Un assistant IA, capable d'exécuter des commandes vocales et de répondre à la voix.</p>
    </div>

    <div class="feature-block" style="background-color: Transparent;">
        <div class="emoji-title" style="display: flex; align-items: center;">
            <div class="feature-emoji" style="font-size: 35px; margin-right: 5px;">🔬</div>
            <h3 style="color: #4f46e5; font-weight: bolder; margin: 0; font-size: 1.5em;">Vision</h3>
        </div>
        <p style="color: #9ca3af; font-weight: bold;">Un assistant IA doté de capacités de reconnaissance d'images, répondant aux questions basées sur le contenu visuel.</p>
    </div>

    <div class="feature-block" style="background-color: Transparent;">
        <div class="emoji-title" style="display: flex; align-items: center;">
            <div class="feature-emoji" style="font-size: 35px; margin-right: 5px;">🖼️</div>
            <h3 style="color: #4f46e5; font-weight: bolder; margin: 0; font-size: 1.5em;">SDXL</h3>
        </div>
        <p style="color: #9ca3af; font-weight: bold;">Modèle Stable diffusion, génère des images haute qualité pour des créations artistiques et graphiques sophistiquées.</p>
    </div>

    <div class="feature-block" style="background-color: Transparent;">
        <div class="emoji-title" style="display: flex; align-items: center;">
            <div class="feature-emoji" style="font-size: 35px; margin-right: 5px;">🖼️</div>
            <h3 style="color: #4f46e5; font-weight: bolder; margin: 0; font-size: 1.5em;">SD3.x</h3>
        </div>
        <p style="color: #9ca3af; font-weight: bold;">Nouveaux modèles avancés de Stable diffusion, génère des images haute qualité pour des créations avec support du texte.</p>
    </div>

    <div class="feature-block" style="background-color: Transparent;">
        <div class="emoji-title" style="display: flex; align-items: center;">
            <div class="feature-emoji" style="font-size: 35px; margin-right: 5px;">🎴</div>
            <h3 style="color: #4f46e5; font-weight: bolder; margin: 0; font-size: 1.5em;">Flux.1</h3>
        </div>
        <p style="color: #9ca3af; font-weight: bold;">Modèle avancé de diffusion de Black Forest Labs,  offrant une qualité d’image exceptionnelle et une précision élevée.</p>
    </div>

    <div class="feature-block" style="background-color: Transparent;">
        <div class="emoji-title" style="display: flex; align-items: center;">
            <div class="feature-emoji" style="font-size: 35px; margin-right: 5px;">🎴</div>
            <h3 style="color: #4f46e5; font-weight: bolder; margin: 0; font-size: 1.5em;">FaceSwap</h3>
        </div>
        <p style="color: #9ca3af; font-weight: bold;">Modèle avancé de diffusion conçu pour échanger les visages de différentes personnes dans des photos.</p>
    </div>

    <div class="feature-block" style="background-color: Transparent;">
        <div class="emoji-title" style="display: flex; align-items: center;">
            <div class="feature-emoji" style="font-size: 35px; margin-right: 5px;">🌐</div>
            <h3 style="color: #4f46e5; font-weight: bolder; margin: 0; font-size: 1.5em;">Ressources</h3>
        </div>
        <p style="color: #9ca3af; font-weight: bold;">Liens vers les pages de support et de téléchargements des modèles IA.</p>
    </div>
</div>
"""

# Fonction pour importer dynamiquement des modules
def import_module_from_path(module_name, module_path):
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        print(Fore.GREEN + f"Successfully imported {module_name}")
        return module
    except FileNotFoundError as e:
        print(Fore.RED + f"File not found for {module_name}: {str(e)}")
    except ImportError as e:
        print(Fore.RED + f"Import error for {module_name}: {str(e)}")
    except Exception as e:
        print(Fore.RED + f"Unexpected error importing {module_name}: {str(e)}")
    return None

# Chemins pour les modules
base_path = os.path.expandvars(r"%userprofile%\eXtraLlama")

app_paths = {
    "📄Languages": {
        "💬 Chatbot": os.path.join(base_path, "Python\Chatbot.py"),
        "📄 PDF": os.path.join(base_path, "Python\PDF.py"),
        "🎙️ Transcript": os.path.join(base_path, "Python\Transcript.py"),
        "📢 Commandes": os.path.join(base_path, "Python\Commandes.py"),
    },
    "🔬 Vision": {
        "🔬 Vision": os.path.join(base_path, "Python\Vision.py"),
    },
    "🖼️ Diffusions": {
        "🖼️ SDXL": os.path.join(base_path, "Python\SDXL.py"),
        "🖼️ SD3.x": os.path.join(base_path, "Python\SD3.x.py"),
        "🎴 Flux.1": os.path.join(base_path, "Python\Flux1.py"),
        "🎴 FaceSwap": os.path.join(base_path, "Python\FaceSwap.py"),
    },
    "🌐 Ressources": {
        "🌐 Ressources & Support": os.path.join(base_path, "Python\Ressources.py"),
    }
}

# Importation des modules
print("\nStarting tabs import...\n")
imported_apps = {}
for category, modules in app_paths.items():
    imported_apps[category] = {}
    for name, path in modules.items():
        module = import_module_from_path(name, path)
        if module and hasattr(module, 'launch_interface'):
            imported_apps[category][name] = module
        else:
            print(Fore.RED + f"Warning: Module {name} not imported or does not have a launch_interface function")
print("\nLaunching tabs on portal...\n")

# Fonction pour démarrer le portail sans duplication des blocs
def launch_portal():
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(spacing_size="sm", text_size="lg")) as interface:
        # Titre principal et conteneur de fonctionnalités en grille
        with gr.Tabs():
            with gr.Tab("🎯 Accueil"):
                gr.HTML(feature_blocks_html)

            # Ajout des onglets et sous-onglets pour chaque module sans doublons
            for category, modules in imported_apps.items():
                with gr.Tab(f"{category}"):
                    with gr.Tabs():
                        for name, module in modules.items():
                            with gr.Tab(name):
                                try:
                                    module.launch_interface()
                                    print(Fore.GREEN + f"Interface for {name} launched successfully")
                                except Exception as e:
                                    print(Fore.RED + f"Error launching {name}: {str(e)}")
                                    gr.Markdown(f"Erreur lors du lancement de {name}: {str(e)}")
        print("\nStarting local server...\n")

    return interface

# Démarrer l'application Gradio
if __name__ == "__main__":
    portal_interface = launch_portal()
    portal_interface.launch(server_name="0.0.0.0", server_port=34000, share=True)