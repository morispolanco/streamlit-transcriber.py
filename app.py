import streamlit as st
import openai
import os
from io import BytesIO

# Configuración de la API (se recomienda guardar la clave en st.secrets)
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Transcriptor de Audio", page_icon="🎙️", layout="centered")

st.title("🎙️ Transcriptor de Audio en Español")
st.write("Sube un archivo de audio (mp3, wav, m4a) y obtén la transcripción en texto.")

# Subida de archivo
audio_file = st.file_uploader("Selecciona un archivo de audio", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    st.audio(audio_file, format="audio/mp3")

    if st.button("Transcribir"):
        with st.spinner("Transcribiendo el audio, por favor espera..."):
            # Convertimos el archivo a un formato que acepte la API
            audio_bytes = audio_file.read()
            audio_stream = BytesIO(audio_bytes)
            audio_stream.name = audio_file.name

            try:
                # Llamada a la API de Whisper
                transcript = openai.Audio.transcriptions.create(
                    model="gpt-4o-transcribe",  # O puedes usar "whisper-1"
                    file=audio_stream,
                    language="es"
                )

                texto = transcript["text"]
                st.success("✅ Transcripción completada")
                st.text_area("Texto transcrito:", texto, height=300)

                # Botón de descarga
                st.download_button(
                    label="⬇️ Descargar Transcripción",
                    data=texto,
                    file_name="transcripcion.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"❌ Error al transcribir: {str(e)}")
