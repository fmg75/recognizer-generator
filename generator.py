from github import Github
import os
import shutil

Name_Proyect = "recognizer_base"
# Autenticarse con la API de GitHub
g = Github("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")

# Obtener el repositorio
repo = g.get_repo("fmg75/Reconocedor_PoH")

# Crear una nueva carpeta para el nuevo proyecto
if not os.path.exists(Name_Proyect):
    os.makedirs(Name_Proyect)

# Descargar los archivos requeridos y guardarlos en la carpeta nueva
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.name in ["app.py", "requirements.txt", "utils.py"]:
        file = repo.get_contents(file_content.path)
        with open(os.path.join(Name_Proyect, file_content.name), "w") as f:
            f.write(file.decoded_content.decode("utf-8"))

# Agregar el archivo caracterisiticas.pkl al nuevo proyecto
ruta_local_archivo = "caracteristicas.pkl"
ruta_destino_archivo = os.path.join(Name_Proyect, "caracteristicas.pkl")
shutil.copy(ruta_local_archivo, ruta_destino_archivo)


# Obtener el repositorio creado anteriormente
repo = g.get_user().get_repo(Name_Proyect)

# Subir los archivos al repositorio
commit_message = "Agregar archivos requeridos"
repo.create_file("app.py", commit_message, "Contenido de app.py")
repo.create_file("requirements.txt", commit_message, "Contenido de requirements.txt")
repo.create_file("utils.py", commit_message, "Contenido de utils.py")
repo.create_file(
    "caracteristicas.pkl",
    "Agregar caracteristicas.pkl",
    "Contenido de caracteristicas.pkl",
)

# Sincronizar con Visual Studio Code
os.system("code .")
