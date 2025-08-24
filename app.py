import streamlit as st
import whisper
import os
import tempfile
from typing import Optional

# Configuración de la página
st.set_page_config(
    page_title="Transcriptor de Audio en Español",
    page_icon="🎤",
    layout="wide"
)

# Título y descripción
st.title("🎤 Transcriptor de Audio en Español")
st.markdown("""
Esta aplicación transcribe archivos de audio en español utilizando el modelo Whisper de OpenAI.
Simplemente carga tu archivo de audio y espera la transcripción.
""")

# Barra lateral para configuración
st.sidebar.header("Configuración")

# Selector de modelo
model_options = {
    "tiny": "Tiny (39MB, más rápido, menos preciso)",
    "base": "Base (74MB, equilibrio velocidad-precisión)",
    "small": "Small (244MB, buena precisión)",
    "medium": "Medium (769MB, alta precisión)",
    "large": "Large (1550MB, máxima precisión)"
}

selected_model = st.sidebar.selectbox(
    "Selecciona un modelo de Whisper:",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x],
    index=1  # Por defecto "base"
)

# Información sobre el modelo
st.sidebar.info(f"""
Modelo seleccionado: {selected_model}

Tamaños y características:
- Tiny: Rápido pero menos preciso
- Base: Buen equilibrio
- Small: Buena precisión
- Medium: Alta precisión
- Large: Máxima precisión (requiere más recursos)
""")

# Función para cargar el modelo
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_name: str):
    """
    Carga el modelo de Whisper y lo cachea para no tener que recargarlo en cada ejecución.
    """
    with st.spinner(f"Cargando modelo {model_name}... Esto puede tardar unos minutos."):
        model = whisper.load_model(model_name)
    return model

# Área principal para cargar el archivo
audio_file = st.file_uploader(
    "Sube un archivo de audio (MP3, WAV, M4A, etc.):",
    type=["mp3", "wav", "m4a", "ogg", "flac", "aac"]
)

# Botón para transcribir
transcribe_button = st.button("Transcribir Audio", disabled=not audio_file)

# Procesamiento y transcripción
if transcribe_button and audio_file:
    # Crear un archivo temporal para guardar el audio subido
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(audio_file.read())
        temp_file_path = temp_file.name

    try:
        # Cargar el modelo
        model = load_whisper_model(selected_model)
        
        # Mostrar mensaje de procesamiento
        with st.spinner("Transcribiendo audio... Por favor espera."):
            # Transcribir el audio
            result = model.transcribe(
                temp_file_path,
                language="es",
                verbose=False,  # Cambiar a True para ver detalles del proceso
                fp16=False  # False para compatibilidad con más dispositivos
            )
        
        # Mostrar la transcripción
        st.subheader("Transcripción:")
        st.text_area("Texto transcrito:", value=result["text"], height=300)
        
        # Opción para descargar la transcripción
        st.download_button(
            label="Descargar transcripción como archivo de texto",
            data=result["text"],
            file_name=f"transcripcion_{os.path.splitext(audio_file.name)[0]}.txt",
            mime="text/plain"
        )
        
        # Mostrar información adicional si está disponible
        if "segments" in result:
            with st.expander("Ver segmentos con marcas de tiempo"):
                segments_df = []
                for segment in result["segments"]:
                    segments_df.append({
                        "Inicio": f"{segment['start']:.2f}s",
                        "Fin": f"{segment['end']:.2f}s",
                        "Texto": segment["text"]
                    })
                
                st.dataframe(segments_df)
    
    except Exception as e:
        st.error(f"Ocurrió un error durante la transcripción: {str(e)}")
    
    finally:
        # Eliminar el archivo temporal
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# Instrucciones adicionales
with st.expander("Instrucciones de uso"):
    st.markdown("""
    1. Selecciona el modelo de Whisper que deseas usar en la barra lateral.
       - Modelos más grandes son más precisos pero requieren más tiempo y recursos.
    2. Sube un archivo de audio en español usando el botón de arriba.
       - Formatos soportados: MP3, WAV, M4A, OGG, FLAC, AAC.
    3. Haz clic en el botón "Transcribir Audio".
    4. Espera a que se complete la transcripción.
    5. Puedes descargar la transcripción como un archivo de texto.
    
    **Nota:** La primera vez que uses un modelo, este se descargará, lo que puede tardar varios minutos dependiendo del tamaño del modelo y tu conexión a internet.
    """)

# Información sobre la tecnología utilizada
st.markdown("---")
st.markdown("""
Esta aplicación utiliza [Whisper](https://openai.com/research/whisper), un modelo de reconocimiento de voz desarrollado por OpenAI.
Whisper ofrece un rendimiento robusto en la transcripción de audio en múltiples idiomas, incluido el español.
""")

