import streamlit as st
import whisper
import os
from tempfile import NamedTemporaryFile

# Título de la app
st.title("Transcriptor de Audio en Español")
st.write("Sube un archivo de audio y obtén su transcripción automática en español.")

# Cargar modelo de Whisper
@st.cache_resource
def cargar_modelo():
    return whisper.load_model("base")

modelo = cargar_modelo()

# Subir archivo
archivo_audio = st.file_uploader("Sube un archivo de audio", type=["mp3", "wav", "m4a", "flac", "ogg"])

if archivo_audio is not None:
    with st.spinner("Transcribiendo..."):
        # Guardar temporalmente el archivo
        with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(archivo_audio.read())
            tmp_path = tmp_file.name

        # Transcribir
        resultado = modelo.transcribe(tmp_path, language="es")
        texto_transcrito = resultado["text"]

        # Mostrar resultado
        st.subheader("Transcripción:")
        st.text_area("Texto transcrito", texto_transcrito, height=300)

        # Botón para descargar
        st.download_button(
            label="Descargar transcripción (.txt)",
            data=texto_transcrito,
            file_name="transcripcion.txt",
            mime="text/plain"
        )

        # Limpiar archivo temporal
        os.remove(tmp_path)
