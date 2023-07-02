import os
import io
import streamlit as st
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
import uuid
import base64

# Generar un ID único utilizando uuid
unique_id = str(uuid.uuid4())[:8]


class FaceNetModels:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained="vggface2").eval()
        self.mtcnn = MTCNN(min_face_size=50, keep_all=False)
        self.caracteristicas = None

    def load_caracteristicas(self, filename):
        with open(filename, "rb") as f:
            self.caracteristicas = pickle.load(f)

    def embedding(self, img_tensor):
        img_embedding = self.model(img_tensor.unsqueeze(0))
        return img_embedding

    def Distancia(self, img_embedding):
        distances = [
            (label, torch.dist(emb, img_embedding))
            for label, emb in self.caracteristicas.items()
        ]
        sorted_distances = sorted(distances, key=lambda x: x[1])
        return sorted_distances[0][0], sorted_distances[0][1].item()

    def extract_embeddings(self, uploaded_files):
        embeddings_list = []
        labels = []
        no_process_images = []

        for uploaded_file in uploaded_files:
            img_path = uploaded_file.name
            img = Image.open(uploaded_file)
            img = img.convert("RGB")
            label = os.path.splitext(uploaded_file.name)[0]
            face = self.mtcnn(img)

            if face is None:
                no_process_images.append(uploaded_file.name)
                continue

            embeddings_list.append(self.model(face.unsqueeze(0)))
            labels.append(label)

        self.caracteristicas = dict(zip(labels, embeddings_list))

        st.write(f"Se procesaron {len(embeddings_list)} imágenes.")

        if no_process_images:
            st.warning(f"No se pudieron procesar {len(no_process_images)} imágenes.")

        return self.caracteristicas


def run_feature_extraction(uploaded_files):
    _models = FaceNetModels()
    if st.button("Extraer características"):
        try:
            caracteristicas = _models.extract_embeddings(uploaded_files)
            filename = f"feature_{unique_id}.pkl"

            with open(filename, "wb") as f:
                pickle.dump(caracteristicas, f)

            st.write(f"Se guardaron las características en: {filename}")

            return filename

        except Exception as e:
            st.error("Ocurrió un error. Detalles: " + str(e))


def upload_and_process_image(uploaded_file, pkl_file):
    try:
        _models = FaceNetModels()
        _models.load_caracteristicas(pkl_file)

        img = Image.open(io.BytesIO(uploaded_file.read()))

        if img.format == "PNG":
            jpg_io = io.BytesIO()
            img = img.convert("RGB")
            img.save(jpg_io, format="JPEG")
            jpg_io.seek(0)
            img = Image.open(jpg_io)

        image_embedding = _models.embedding(_models.mtcnn(img))

        result = _models.Distancia(image_embedding)
        if result:
            label, distance = result
            st.write("La imagen cargada puede ser de:", label)
            st.write("Distancia Euclidiana: ", round(distance, 4))
            show_recognized_face(label, os.path.dirname(uploaded_file.name))
        else:
            st.write(
                "Algo falló con la imagen proporcionada, intenta con otra!!"
                + "Verifica si la ruta del diccionario de Caracteristicas es correcta "
                + "O si el mismo a sido generado previamente"
            )

    except Exception as e:
        print("Error en upload_and_process_image:", str(e))
        return None


def show_recognized_face(label, data_dir):
    img_files = os.listdir(data_dir)
    for img_file in img_files:
        img_path = os.path.join(data_dir, img_file)
        img_label = os.path.splitext(img_file)[0]
        if img_label == label:
            image = Image.open(img_path)
            st.image(image, caption="Imagen del rostro reconocido", width=200)
            return

    st.write("No se encontró la imagen correspondiente al rostro reconocido.")


# Crear una barra lateral para seleccionar la página
page = st.sidebar.selectbox(
    "Seleccione una página", ["Extracción de Características", "Reconocedor de Rostros"]
)

# Verificar la página seleccionada y mostrar el contenido correspondiente
if page == "Reconocedor de Rostros":
    # Código del segundo archivo (actual)
    st.markdown(
        "<h1 style='color: green;'>RECONOCEDOR DE ROSTROS</h1>", unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader(
        "Subir la imagen de un Rostro",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image, caption="Imagen subida ", width=200)
        pkl_file = run_feature_extraction(uploaded_file)
        if pkl_file is not None:
            upload_and_process_image(uploaded_file, pkl_file)

elif page == "Extracción de Características":
    # Código del primer archivo
    st.markdown(
        "<h1 style='color: green;'>EXTRACCION CARACTERISTICAS</h1>",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Subir imágenes", accept_multiple_files=True, type=["jpg", "jpeg", "png"]
    )

    pkl_file = run_feature_extraction(uploaded_files)
    if pkl_file is not None:
        st.write(f"Ruta del archivo .pkl generado: {pkl_file}")
