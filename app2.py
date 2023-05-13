import streamlit as st
import os
from PIL import Image
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
import uuid

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
    filename = os.path.join(data_dir, f"caracteristicas_{unique_id}.pkl")
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


# Crear los elementos de la interfaz de usuario
st.title("Extractor de características faciales")
data_dir = st.text_input("Ingrese la ruta de la carpeta con las imágenes:")

# Verificar si se ingresó una ruta de carpeta válida
if data_dir and os.path.isdir(data_dir):
    # Crear el botón para iniciar el proceso de extracción de características
    if st.button("Extraer características"):
        save_dir = os.path.abspath(data_dir)
        extract_embeddings(data_dir, save_dir)
        st.success("Se extrajeron las características de las imágenes.")
else:
    st.warning("Ingrese una ruta de carpeta válida para continuar.")
