import streamlit as st
from openai import OpenAI
import tempfile

# Configuración del cliente
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Transcriptor de Audio", page_icon="🎙️", layout="centered")

st.title("🎙️ Transcriptor de Audio en Español")
st.write("Sube un archivo de audio (mp3, wav, m4a) y obtén la transcripción en texto.")

# Subida de archivo
audio_file = st.file_uploader("Selecciona un archivo de audio", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    st.audio(audio_file, format="audio/mp3")

    if st.button("Transcribir"):
        with st.spinner("Transcribiendo el audio, por favor espera..."):
            try:
                # Guardar archivo temporalmente en disco (mejor compatibilidad con OpenAI)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name

                # Abrir archivo en modo binario y enviarlo a la API
                with open(tmp_path, "rb") as f:
                    transcript = client.audio.transcriptions.create(
                        model="gpt-4o-transcribe",  # o "whisper-1"
                        file=f,
                        language="es"
                    )

                texto = transcript.text
                st.success("✅ Transcripción completada")
                st.text_area("Texto transcrito:", texto, height=300)

                st.download_button(
                    label="⬇️ Descargar Transcripción",
                    data=texto,
                    file_name="transcripcion.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"❌ Error al transcribir: {str(e)}")
