import streamlit as st
import os
from PIL import Image
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
import uuid
import os
import requests
import base64


def send_petition(caracteristicas):
    unique_id = str(uuid.uuid4())
    access_token = "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"
    repo_owner = "fmg75"
    repo_name = "recognizer-generator"
    branch = "master"

    file_contents = pickle.dumps(caracteristicas)

    #  Configurar la URL de la API de GitHub
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/caracteristicas_{unique_id}.pkl"

    # Configurar los headers de la petición
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {access_token}",
    }

    # Configurar el cuerpo de la petición
    data = {
        "message": f"Subiendo archivo caracteristicas",
        "content": base64.b64encode(file_contents).decode(),
        "branch": branch,
    }

    # Enviar la petición a través de la API de GitHub
    response = requests.put(api_url, headers=headers, json=data)

    # Mostrar la respuesta del servidor
    if response.status_code == 201:
        print(f"Contenido de variable 'caracteristicas' subido con éxito.")
    else:
        print(
            f"Error al subir el contenido de variable 'caracteristicas': {response.text}"
        )


def extract_embeddings(data_dir):
    # Cargar el modelo de reconocimiento facial y el detector de caras
    model = InceptionResnetV1(pretrained="vggface2").eval()
    mtcnn = MTCNN(min_face_size=50, keep_all=False)

    # inicio = 0
    # fin = 1000
    embeddings_list = []
    labels = []
    no_process_dir = os.path.join(data_dir, "no_process")
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
    send_petition(caracteristicas)


# Crear los elementos de la interfaz de usuario
st.title("Extractor de características faciales")
data_dir = st.text_input("Ingrese la ruta de la carpeta con las imágenes:")

# Verificar si se ingresó una ruta de carpeta válida
if data_dir and os.path.isdir(data_dir):
    # Crear el botón para iniciar el proceso de extracción de características
    if st.button("Extraer características"):
        extract_embeddings(data_dir)
        st.success("Se extrajeron las características de las imágenes.")
else:
    st.warning("Ingrese una ruta de carpeta válida para continuar.")
