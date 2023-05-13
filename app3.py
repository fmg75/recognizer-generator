import streamlit as st
import os
from PIL import Image
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
import uuid
import os
import requests
import base64


# Generar un ID único utilizando uuid
unique_id = str(uuid.uuid4())


def extract_embeddings(data_dir, save_dir):
    # Cargar el modelo de reconocimiento facial y el detector de caras
    model = InceptionResnetV1(pretrained="vggface2").eval()
    mtcnn = MTCNN(min_face_size=50, keep_all=False)

    # inicio = 0
    # fin = 1000
    embeddings_list = []
    labels = []
    no_process_dir = os.path.join(save_dir, "no_process")
    os.makedirs(no_process_dir, exist_ok=True)

    # img_files = os.listdir(data_dir)[inicio:fin]
    img_files = os.listdir(data_dir)
    for img_file in img_files:
        img_path = os.path.join(data_dir, img_file)
        # check the file is an image or not
        if os.path.splitext(img_path)[1].lower() in (".jpg", ".jpeg", ".png"):
            img = Image.open(img_path)
            img = img.convert("RGB")
            # Obtener el nombre del archivo de la imagen actual
            label = os.path.splitext(os.path.basename(img_path))[0]
            face = mtcnn(img)
            if face is None:
                img.save(
                    os.path.join(no_process_dir, os.path.splitext(img_file)[0] + ".png")
                )
                continue
            embeddings_list.append(model(face.unsqueeze(0)))
            labels.append(label)

    caracteristicas = dict(zip(labels, embeddings_list))

    # Agregar el ID al nombre del archivo

    # # Generar un ID único utilizando uuid
    # unique_id = str(uuid.uuid4())

    # filename = os.path.join(data_dir, f"caracteristicas_{unique_id}.pkl")
    filename = os.path.join(data_dir, f"caracteristicas_1.pkl")
    # Serializar la variable caracteristicas y guardarla en el archivo con el nombre generado
    with open(filename, "wb") as f:
        pickle.dump(caracteristicas, f)

    # Mostrar la cantidad de imágenes procesadas y no procesadas
    st.write(f"Se procesaron {len(embeddings_list)} imágenes.")
    no_process_files = os.listdir(no_process_dir)
    if no_process_files:
        st.warning(
            f"No se pudieron procesar {len(no_process_files)} imágenes. "
            f"Revise la carpeta '{no_process_dir}' para ver los archivos no procesados."
        )
    return filename


def send_petition(filename):
    file_path = "caracteristicas_1.pkl"
    # file_path = "caracteristicas/caracteristicas_" + unique_id + ".pkl"
    repo_owner = "fmg75"
    repo_name = "recognizer-generator"
    branch = "master"
    access_token = "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"
    filename_ = filename
    # Leer el archivo y convertirlo a bytes
    with open(filename_, "rb") as f:
        file_contents = f.read()

    # Codificar los bytes a base64
    file_contents_encoded = base64.b64encode(file_contents).decode()

    # Configura la URL de la API de GitHub
    api_url = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
    )

    # Configura los headers de la petición
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    # Crea el cuerpo de la petición
    data = {
        "message": f"Subiendo archivo {file_path}",
        "content": file_contents_encoded,
        "branch": branch,
    }

    # Realiza la petición
    response = requests.put(api_url, headers=headers, json=data)

    # Imprime el resultado
    print(response.json())
    print(api_url)
    print(filename_)


# Crear los elementos de la interfaz de usuario
st.title("Extractor de características faciales")
data_dir = st.text_input("Ingrese la ruta de la carpeta con las imágenes:")
# file_path = os.path.join(data_dir, f"caracteristicas_{unique_id}.pkl")
# Verificar si se ingresó una ruta de carpeta válida
if data_dir and os.path.isdir(data_dir):
    # Crear el botón para iniciar el proceso de extracción de características
    if st.button("Extraer características"):
        save_dir = os.path.abspath(data_dir)
        # extract_embeddings(data_dir, save_dir)
        send_petition(extract_embeddings(data_dir, save_dir))
        st.success("Se extrajeron las características de las imágenes.")
else:
    st.warning("Ingrese una ruta de carpeta válida para continuar.")


# import requests
# import base64

# # Configura la información del repositorio
# repo_owner = "tu_usuario_en_github"
# repo_name = "nombre_de_tu_repositorio"
# branch_name = "nombre_de_la_rama"

# # Configura la ruta y el nombre del archivo
# file_path = "carpeta/dentro/del/repositorio/archivo.txt"

# # Lee el contenido del archivo y conviértelo a base64
# with open("C:/Users/tu_usuario/Escritorio/archivo.txt", "rb") as f:
#     file_content = f.read()
# file_content_base64 = base64.b64encode(file_content).decode("utf-8")

# # Crea el objeto de solicitud de API
# api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
# headers = {"Authorization": f"token {tu_token_de_autorizacion_github}"}
# payload = {
#     "message": "Agregar archivo desde mi escritorio",
#     "content": file_content_base64,
#     "branch": branch_name,
# }

# # Envía la solicitud para subir el archivo
# response = requests.put(api_url, headers=headers, json=payload)

# # Imprime el resultado de la solicitud
# print(response.json())
