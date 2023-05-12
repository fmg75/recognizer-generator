from github import Github
import os


# Define el nombre del repositorio a crear
Name_Proyect = "recognizer_base"

# Autenticarse con la API de GitHub
g = Github("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")

# Verificar si la repo ya existe
try:
    repo = g.get_user().get_repo(Name_Proyect)
    print(f"La repo '{Name_Proyect}' ya existe. No se creará una nueva repo.")
except:
    # Crear una nueva repo
    repo = g.get_user().create_repo(Name_Proyect)
    print(f"Se creó una nueva repo llamada '{Name_Proyect}'.")

# Clonar el repositorio recién creado
repo = g.get_user().get_repo(Name_Proyect)

# Obtener los archivos necesarios del repositorio original
original_repo = g.get_repo("https://github.com/fmg75/Reconocedor_PoH.git")
contents = original_repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir":
        contents.extend(original_repo.get_contents(file_content.path))
        os.makedirs(os.path.join(Name_Proyect, file_content.path))
    else:
        if file_content.name in [
            "app.py",
            "requirements.txt",
            "utils.py",
            "caracteristicas.pkl",
        ]:
            file = original_repo.get_contents(file_content.path)
            with open(os.path.join(Name_Proyect, file_content.name), "w") as f:
                f.write(file.decoded_content.decode("utf-8"))

# Agregar los archivos al repositorio nuevo
commit_message = "Initial commit"
repo.create_file("app.py", commit_message, open("app.py", "rb").read())
repo.create_file(
    "requirements.txt", commit_message, open("requirements.txt", "rb").read()
)
repo.create_file("utils.py", commit_message, open("utils.py", "rb").read())
repo.create_file(
    "caracteristicas.pkl", commit_message, open("caracteristicas.pkl", "rb").read()
)
