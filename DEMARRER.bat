@echo off
setlocal

:: Chemin vers le script Python
set MODIFY_SCRIPT=%~dp0Python\modify_main.py

:: Exécuter le script Python pour modifier main.py
python %MODIFY_SCRIPT%
if %ERRORLEVEL% NEQ 0 (
    echo Erreur : Le script Python n'a pas pu modifier main.py.
    pause
    exit /b
)

:: Créer un raccourci sur le bureau pour ComfyUI
cd /d %~dp0
set SHORTCUTPATH="%userprofile%\Desktop\"ComfyUI".url"
echo [InternetShortcut] > "%SHORTCUTPATH%"
echo URL="http://localhost:8188/" >> "%SHORTCUTPATH%"
echo IconFile="%CD%\Fichiers\ComfyUI.ico" >> "%SHORTCUTPATH%"
echo IconIndex=0 >> "%SHORTCUTPATH%"
cls

:: Créer un raccourci sur le bureau pour le dossier des diffusions
cd /d %~dp0
set SHORTCUTPATH="%userprofile%\Desktop\"Diffusions ComfyUI".url"
echo [InternetShortcut] > "%SHORTCUTPATH%"
echo URL="%userprofile%\ComfyUI\ComfyUI\output" >> "%SHORTCUTPATH%"
echo IconFile="%CD%\Fichiers\Diffusion.ico" >> "%SHORTCUTPATH%"
echo IconIndex=0 >> "%SHORTCUTPATH%"
cls

:: Créer un raccourci sur le bureau pour eXtra Llama
cd /d %~dp0
set SHORTCUTPATH="%userprofile%\Desktop\"  Portail IA     eXtra Llama".url"
echo [InternetShortcut] > "%SHORTCUTPATH%"
echo IconFile="%CD%\Fichiers\eXtraLlama.ico" >> "%SHORTCUTPATH%"
echo IconIndex=0 >> "%SHORTCUTPATH%"
echo URL="%CD%\DEMARRER.bat" >> "%SHORTCUTPATH%"
cls

:: Vérifier si ComfyUI est déjà en cours d'exécution
tasklist /FI "IMAGENAME eq python.exe" | find /I "python.exe" >nul
if "%ERRORLEVEL%" NEQ "0" (
    cd /d %userprofile%\ComfyUI
    start "ComfyUI" /min cmd /c "%userprofile%\ComfyUI\run_nvidia_gpu.bat & exit"
)

:: Vérifier si ollama est déjà en cours d'exécution
tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul
if "%ERRORLEVEL%" NEQ "0" (
    cd /d %~dp0
    start "ollama serve" /min cmd /c "ollama serve & exit"
)

:: Créer l'environnement Python et installer les dépendances
cd /d %~dp0
if not exist .venv (
    python.exe -m venv .venv
)
call .\.venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r .\requirements.txt

:: Vérifier si Portail.py est déjà en cours d'exécution et le fermer s'il l'est
tasklist /FI "IMAGENAME eq python.exe" | find /I "Python\Portail.py" >nul
if "%ERRORLEVEL%" EQU "0" (
    :: Fermer Portail.py
    taskkill /FI "IMAGENAME eq python.exe" /F
)

:: Changer de répertoire vers le dossier Python et démarrer Portail.py
cd /d %~dp0Python
start "Portail.py" /min cmd /c "call ..\.venv\Scripts\activate && python Portail.py & exit"

:: Attendre que le serveur démarre
timeout /t 10 /nobreak >nul

:: Ouvrir le navigateur avec l'URL (une seule fois)
start "" http://localhost:34000/

