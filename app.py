import streamlit as st
import os
import tempfile
import sys
import subprocess
import shutil

# Configuración de la página
st.set_page_config(
    page_title="Transcriptor de Audio en Español",
    page_icon="🎤",
    layout="wide"
)

# Función para verificar si ffmpeg está instalado
def check_ffmpeg():
    try:
        # Verificar si ffmpeg está en el PATH
        result = subprocess.run(["ffmpeg", "-version"], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True)
        if result.returncode == 0:
            return True
        return False
    except FileNotFoundError:
        return False

# Función para instalar ffmpeg según el sistema operativo
def install_ffmpeg_instructions():
    st.error("❌ FFmpeg no está instalado en tu sistema. Es necesario para procesar archivos de audio.")
    
    st.markdown("### Instrucciones de instalación según tu sistema operativo:")
    
    st.markdown("""
    #### **Windows:**
    1. Descarga FFmpeg desde https://ffmpeg.org/download.html
    2. Extrae el archivo descargado
    3. Agrega la carpeta `bin` a tu PATH del sistema:
       - Presiona Win + R, escribe `sysdm.cpl`
       - Ve a la pestaña "Opciones avanzadas"
       - Haz clic en "Variables de entorno"
       - En "Variables del sistema", busca "Path" y haz clic en "Editar"
       - Agrega la ruta a la carpeta `bin` de FFmpeg (ej: `C:\ffmpeg\bin`)
    4. Reinicia tu computadora y vuelve a intentar
    """)
    
    st.markdown("""
    #### **macOS:**
    ```bash
    # Usando Homebrew (recomendado)
    brew install ffmpeg
    
    # O usando MacPorts
    sudo port install ffmpeg
    ```
    """)
    
    st.markdown("""
    #### **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt update
    sudo apt install ffmpeg
    ```
    
    #### **Linux (Fedora/CentOS):**
    ```bash
    sudo dnf install ffmpeg
    ```
    """)

# Verificar ffmpeg al inicio
if not check_ffmpeg():
    install_ffmpeg_instructions()
    st.stop()

# Ahora continuamos con el resto de la aplicación
try:
    import whisper
    st.success("✅ Whisper está instalado correctamente")
except ImportError:
    st.error("❌ Whisper no está instalado. Por favor, instala las dependencias necesarias.")
    st.code("""
    # Para instalar Whisper y sus dependencias:
    pip install openai-whisper torch
    """)
    st.stop()

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
    try:
        with st.spinner(f"Cargando modelo {model_name}... Esto puede tardar unos minutos."):
            model = whisper.load_model(model_name)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

# Función para convertir audio usando pydub (alternativa)
def convert_audio_with_pydub(input_path, output_path):
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return True
    except ImportError:
        st.error("pydub no está instalado. Intenta instalarlo con: pip install pydub")
        return False
    except Exception as e:
        st.error(f"Error al convertir el audio: {str(e)}")
        return False

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
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as temp_file:
        temp_file.write(audio_file.read())
        temp_file_path = temp_file.name
    
    # Crear un archivo temporal para el audio convertido (si es necesario)
    converted_temp_path = None
    
    try:
        # Cargar el modelo
        model = load_whisper_model(selected_model)
        
        if model is None:
            st.error("No se pudo cargar el modelo. Por favor, intenta con otro modelo o verifica tu instalación.")
            st.stop()
        
        # Mostrar mensaje de procesamiento
        with st.spinner("Transcribiendo audio... Por favor espera."):
            # Intentar transcribir directamente
            try:
                result = model.transcribe(
                    temp_file_path,
                    language="es",
                    verbose=False,
                    fp16=False
                )
            except Exception as e:
                st.warning(f"Error al procesar el audio directamente: {str(e)}")
                st.info("Intentando convertir el audio a WAV...")
                
                # Crear un archivo temporal para la conversión
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as converted_file:
                    converted_temp_path = converted_file.name
                
                # Intentar convertir con pydub
                if convert_audio_with_pydub(temp_file_path, converted_temp_path):
                    result = model.transcribe(
                        converted_temp_path,
                        language="es",
                        verbose=False,
                        fp16=False
                    )
                else:
                    st.error("No se pudo procesar el archivo de audio. Asegúrate de que ffmpeg esté instalado correctamente.")
                    st.stop()
        
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
        # Eliminar los archivos temporales
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if converted_temp_path and os.path.exists(converted_temp_path):
            os.unlink(converted_temp_path)

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

