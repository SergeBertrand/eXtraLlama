import os

# Chemin vers le fichier main.py
main_py_path = os.path.join(os.environ['USERPROFILE'], 'ComfyUI', 'ComfyUI', 'main.py')

# Lire le fichier main.py
try:
    with open(main_py_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
except Exception as e:
    print(f"Erreur lors de la lecture du fichier {main_py_path}: {e}")
    exit(1)

# Modifier la ligne spécifique
modified_lines = []
for line in lines:
    if 'webbrowser.open(f"{scheme}://{address}:{port}")' in line and not line.strip().startswith('#'):
        modified_lines.append('#' + line)
    else:
        modified_lines.append(line)

# Écrire les modifications dans le fichier
try:
    with open(main_py_path, 'w', encoding='utf-8') as file:
        file.writelines(modified_lines)
    print(f"Modification effectuée dans {main_py_path}")
except Exception as e:
    print(f"Erreur lors de l'écriture dans le fichier {main_py_path}: {e}")
    exit(1)
