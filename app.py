import streamlit as st
from PIL import Image
from utils import *

st.markdown(
    "<h1 style='color: green;'>RECONOCEDOR DE ROSTROS</h1>",
    unsafe_allow_html=True,
)


def process_image(path):
    try:
        _models = FaceNetModels()
        img = Image.open(path)
        image_embedding = _models.embedding(_models.mtcnn(img))
        return _models.Distancia(image_embedding)
    except:
        return None


def upload_image():
    uploaded_file = st.file_uploader(
        "Subir la imagen de un Rostro",
        type=["jpg", "png"],
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida ", width=200)
        result = process_image(uploaded_file)
        if result:
            label, distance = result
            st.write("La imagen cargada es de:", label)
            st.write("Distancia Euclidiana: ", round(distance, 4))
        else:
            st.write("Algo falló con la imagen proporcionada, intenta con otra !!")

    # información adicional
    with st.expander("Información adicional"):
        st.write(".....")


# lanza app
upload_image()
