import os
import sqlite3
import datetime
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk
import threading
import requests # Import the requests library for HTTP communication
import json     # Import json for handling JSON data
import time     # For potential delays or loading animations

# Importar la función de predicción del módulo de visión
try:
    from modulo1_vision import predecir_emocion_y_persona
except ImportError:
    messagebox.showerror("Error de Importación",
                          "No se pudo importar 'predecir_emocion_y_persona' del archivo 'modulo1_vision.py'. "
                          "Asegúrate de que el archivo existe y el Módulo 1 se ejecutó para generar 'mejor_modelo.h5'.")
    exit()

# --- Configuración para el SLM Gemma 2B ---
# Define la URL de tu API local de Gemma 2B.
# Si usas Ollama, por defecto suele ser http://localhost:11434/api/generate
# Asegúrate de que tu modelo 'gemma2:2b' está disponible en Ollama.
GEMMA_API_URL = "http://localhost:11434/api/generate"
GEMMA_MODEL_NAME = "gemma2:2b" # El nombre del modelo que tienes en Ollama o tu servidor local

# --- Base de Datos para Conversaciones del Chat ---
DB_NAME = 'chat_history.db'

def setup_database():
    """Configura la tabla de conversaciones en la base de datos."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            speaker TEXT NOT NULL,
            message TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_message(user_id, speaker, message):
    """Guarda un mensaje en la base de datos."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    cursor.execute("INSERT INTO conversations (user_id, timestamp, speaker, message) VALUES (?, ?, ?, ?)",
                   (user_id, timestamp, speaker, message))
    conn.commit()
    conn.close()

def get_conversation_history(user_id, limit=10):
    """Recupera el historial de conversación para un usuario dado."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT speaker, message FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
    history = cursor.fetchall()
    conn.close()
    # Ollama typically expects "user" and "assistant" roles if using the chat endpoint.
    # For the generate endpoint, we'll convert it to a string later.
    formatted_history = []
    for row in reversed(history):
        speaker_role = "user" if row[0] == "user" else "assistant" # Assuming 'model' maps to 'assistant'
        formatted_history.append({"role": speaker_role, "content": row[1]})
    return formatted_history

# --- Configuración de Prompts para diferentes usuarios y emociones ---
user_prompts = {
    'personaA': {
        'initial': "Eres un agente de chat amigable y entusiasta. Estás hablando con Persona A. Haz tus respuestas atractivas y positivas.",
        'alegre': "¡Persona A está feliz! Responde con mensajes alegres y alentadores.",
        'triste': "Persona A parece triste. Ofrece consuelo e intenta animarle suavemente.",
        'pensativo': "Persona A está pensativa. Participa en conversaciones reflexivas e interesantes.",
        'con_ira': "Persona A parece enfadada. Responde con calma e intenta calmar la situación.",
        'cansado': "Persona A parece cansada. Ofrece palabras de consuelo y sugiere descanso.",
        'sorprendido': "¡Persona A está sorprendida! Expresa asombro y curiosidad.",
        'riendo': "¡Persona A se está riendo! Comparte su alegría con respuestas juguetonas y desenfadadas."
    },
    'personaB': {
        'initial': "Eres un agente de chat formal e informativo. Estás hablando con Persona B. Proporciona información concisa y útil.",
        'alegre': "¡Persona B está feliz! Mantén un tono educado pero positivo.",
        'triste': "Persona B parece triste. Ofrece apoyo práctico o un silencio empático.",
        'pensativo': "Persona B está pensativa. Proporciona información fáctica o haz preguntas que inviten a la reflexión.",
        'con_ira': "Persona B parece enfadada. Aborda sus preocupaciones profesionalmente y busca resolver los problemas.",
        'cansado': "Persona B parece cansada. Reconoce su estado y ofrece información concisa.",
        'sorprendido': "¡Persona B está sorprendida! Proporciona contexto o explicaciones.",
        'riendo': "¡Persona B se está riendo! Reconoce su diversión breve y cortésmente."
    }
}

# Variables globales para el estado del chat
current_user = None
current_emotion = None
chat_window = None
chat_display = None
user_entry = None
send_button = None
initial_photo_path_entry = None
initial_photo_button = None
process_initial_button = None
update_emotion_photo_entry = None
update_emotion_button = None
progress_bar = None # New progress bar widget

# --- Función para comunicarse con el SLM local (Gemma 2B) ---
def get_gemma_response(user_message, current_prompt_context_text, conversation_history):
    """
    Envía una solicitud al SLM local (Gemma 2B) y obtiene su respuesta.
    'user_message' es el mensaje que el usuario acaba de escribir.
    'current_prompt_context_text' es la cadena que describe el rol y la emoción.
    'conversation_history' es una lista de dicts con 'role' y 'content'.
    """
    
    # We will build the *entire* prompt for Gemma in a single string,
    # as the /api/generate endpoint works best with this for simpler models.
    
    # Start with the core instructions and persona:
    combined_initial_prompt = (
        f"{user_prompts[current_user]['initial']} "
        f"El usuario actual es '{current_user}'. "
        f"La emoción detectada de {current_user} es '{current_emotion}'. "
        f"Debes adaptar tu tono de conversación para ser '{user_prompts[current_user][current_emotion].lower()}'. "
        "Responde de forma concisa, relevante y siempre en español. "
        "Si es una continuación de una conversación, considera el historial. "
        "Si el usuario está cambiando de emoción, ajusta tu respuesta en consecuencia. "
    )

    full_prompt_string = combined_initial_prompt

    if conversation_history:
        full_prompt_string += "\n\nHistorial de Conversación Reciente:\n"
        for msg in conversation_history:
            # Map 'assistant' back to 'Agente' for readability in the prompt context
            speaker_label = "Tú" if msg["role"] == "user" else "Agente"
            full_prompt_string += f"{speaker_label}: {msg['content']}\n"
    
    # Add the current user's message and prompt Gemma to respond
    full_prompt_string += f"\n\nTu mensaje actual: {user_message}\nAgente: "

    payload = {
        "model": GEMMA_MODEL_NAME,
        "prompt": full_prompt_string, # IMPORTANT CHANGE: Use "prompt" key for /api/generate
        "stream": False,
        "options": {
            "temperature": 0.7,   # Control randomness (0.0-1.0)
            "num_predict": 250,   # Max tokens to generate per response (adjust as needed)
            "top_k": 40,          # Consider top_k most likely next tokens
            "top_p": 0.9          # Nucleus sampling parameter
        }
    }
    
    try:
        response = requests.post(GEMMA_API_URL, json=payload, timeout=180) # Increased timeout
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        
        response_data = response.json()
        
        # IMPORTANT CHANGE: For /api/generate, response is typically in 'response' key
        if 'response' in response_data:
            return response_data['response'].strip()
        else:
            print(f"Respuesta inesperada del SLM: {response_data}")
            return "Error: Formato de respuesta inesperado del SLM local."

    except requests.exceptions.ConnectionError:
        return "Error de conexión: Asegúrate de que el servidor de Gemma 2B está corriendo en la URL especificada (ej. Ollama)."
    except requests.exceptions.Timeout:
        return "Error de tiempo de espera: El servidor de Gemma 2B tardó demasiado en responder. Intenta de nuevo o verifica tu instalación."
    except requests.exceptions.RequestException as e:
        return f"Error al comunicarse con el SLM local: {e}. Asegúrate de que el modelo '{GEMMA_MODEL_NAME}' está descargado y funcionando en Ollama."
    except json.JSONDecodeError:
        return "Error: No se pudo decodificar la respuesta JSON del servidor del SLM. Puede que la respuesta no sea JSON válida."


# --- Funciones de la GUI ---
def send_message_gui():
    global current_user, current_emotion

    user_input = user_entry.get().strip()
    user_entry.delete(0, tk.END)

    if not user_input:
        return

    display_message("Tú: " + user_input, "user_msg")
    save_message(current_user, "user", user_input)

    user_entry.config(state=tk.DISABLED)
    send_button.config(state=tk.DISABLED)
    progress_bar.start(10) # Start indeterminate progress bar
    chat_window.update_idletasks() # Update GUI immediately

    threading.Thread(target=process_chatbot_response, args=(user_input,)).start()

def process_chatbot_response(user_input):
    global current_user, current_emotion

    try:
        history_for_slm = get_conversation_history(current_user, limit=10)
        
        # The full context string for the SLM is built inside get_gemma_response.
        # We pass the current emotion context as a string.
        bot_response = get_gemma_response(user_input, user_prompts[current_user][current_emotion], history_for_slm)
        
        if bot_response.startswith("Error:"):
             display_message(bot_response, "error_msg")
        else:
            save_message(current_user, "model", bot_response)
            display_message("Agente: " + bot_response, "agent_msg")

    except Exception as e:
        display_message(f"Error inesperado al procesar la respuesta del chatbot: {e}", "error_msg")
    finally:
        progress_bar.stop() # Stop progress bar
        user_entry.config(state=tk.NORMAL)
        send_button.config(state=tk.NORMAL)
        chat_window.update_idletasks() # Update GUI immediately


def display_message(message, tag="default"):
    chat_display.config(state=tk.NORMAL)
    chat_display.insert(tk.END, message + "\n", tag)
    chat_display.config(state=tk.DISABLED)
    chat_display.yview(tk.END) # Auto-scroll to the bottom

def select_initial_photo():
    file_path = filedialog.askopenfilename(
        title="Selecciona tu foto inicial",
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        initial_photo_path_entry.delete(0, tk.END)
        initial_photo_path_entry.insert(0, file_path)

def process_initial_photo():
    global current_user, current_emotion

    photo_path = initial_photo_path_entry.get().strip()
    if not photo_path:
        messagebox.showwarning("Advertencia", "Por favor, selecciona una foto inicial.")
        return

    if not os.path.exists(photo_path):
        messagebox.showerror("Error", f"La ruta de la foto no existe: {photo_path}")
        return

    display_message("Analizando la foto para identificar al usuario y su emoción inicial...", "info_msg")
    initial_photo_button.config(state=tk.DISABLED)
    initial_photo_path_entry.config(state=tk.DISABLED)
    process_initial_button.config(state=tk.DISABLED)
    progress_bar.start(10) # Start progress bar for vision processing
    chat_window.update_idletasks()

    threading.Thread(target=run_initial_detection, args=(photo_path,)).start()

def run_initial_detection(photo_path):
    global current_user, current_emotion

    try:
        person, emotion = predecir_emocion_y_persona(photo_path)

        if person and emotion:
            current_user = person
            current_emotion = emotion
            display_message(f"¡Hola {current_user}! Te veo {current_emotion}. 😊", "system_msg")

            # This initial message serves to "prime" the SLM for its role and the user's emotion.
            # We save it to the database for historical context.
            initial_prime_message = "Estoy listo para iniciar la conversación. ¿Cómo puedo ayudarte o qué tienes en mente?"
            
            # Save this "initial state" as if the user provided context and the agent acknowledged.
            # This helps build the history for subsequent SLM calls.
            # The 'user' speaker for this prompt means it's a piece of context provided by the "user side" of the application.
            save_message(current_user, "user", f"El usuario {current_user} ha sido identificado con la emoción {current_emotion}.")
            save_message(current_user, "model", initial_prime_message) # Save the agent's acknowledgment

            display_message("Agente: " + initial_prime_message, "agent_msg")

            user_entry.config(state=tk.NORMAL)
            send_button.config(state=tk.NORMAL)
            update_emotion_button.config(state=tk.NORMAL)
            update_emotion_photo_entry.config(state=tk.NORMAL)
            display_message("Ahora puedes chatear. ¡Intenta escribir algo! También puedes actualizar tu emoción con una nueva foto en cualquier momento.", "info_msg")

        else:
            messagebox.showerror("Error de Detección", "No se pudo identificar al usuario o su emoción en la foto. Por favor, intenta con otra foto.")
            display_message("No se pudo identificar al usuario en la foto.", "error_msg")
            
    except Exception as e:
        messagebox.showerror("Error de Visión", f"Ocurrió un error al procesar la foto con el módulo de visión: {e}")
        display_message(f"Error al procesar la foto de visión: {e}", "error_msg")
    finally:
        progress_bar.stop() # Stop progress bar
        initial_photo_button.config(state=tk.NORMAL)
        initial_photo_path_entry.config(state=tk.NORMAL)
        process_initial_button.config(state=tk.NORMAL)
        chat_window.update_idletasks()


def select_update_emotion_photo():
    file_path = filedialog.askopenfilename(
        title="Selecciona una foto para actualizar tu emoción",
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        update_emotion_photo_entry.delete(0, tk.END)
        update_emotion_photo_entry.insert(0, file_path)
        # --- RE-ADDED: Call processing function immediately after selecting file ---
        process_update_emotion_photo() 

def process_update_emotion_photo():
    global current_user, current_emotion

    if current_user is None:
        messagebox.showwarning("Advertencia", "Primero debes identificar al usuario con la foto inicial.")
        return

    photo_path = update_emotion_photo_entry.get().strip()
    if not photo_path:
        messagebox.showwarning("Advertencia", "Por favor, selecciona una foto para actualizar la emoción.")
        return

    if not os.path.exists(photo_path):
        messagebox.showerror("Error", f"La ruta de la foto no existe: {photo_path}")
        return

    display_message("Analizando la nueva foto para actualizar tu emoción...", "info_msg")
    update_emotion_button.config(state=tk.DISABLED)
    update_emotion_photo_entry.config(state=tk.DISABLED)
    progress_bar.start(10) # Start progress bar for emotion update
    chat_window.update_idletasks()

    threading.Thread(target=run_emotion_update, args=(photo_path,)).start()

def run_emotion_update(photo_path):
    global current_user, current_emotion

    try:
        _, new_emotion = predecir_emocion_y_persona(photo_path)

        if new_emotion:
            if new_emotion != current_emotion:
                display_message(f"¡Oh, ahora te veo {new_emotion}! Mi tono de conversación se adaptará. 😊", "system_msg")
                old_emotion = current_emotion # Store old emotion for the prompt
                current_emotion = new_emotion
                
                # Formulate a message for the SLM to update its internal state/context
                emotion_update_user_message = (
                    f"Mi emoción acaba de cambiar de '{old_emotion}' a '{new_emotion}'. "
                    "Por favor, ajusta tu tono de conversación para que coincida con mi nueva emoción. "
                    f"Recuerda que ahora me siento {current_emotion}."
                )
                
                save_message(current_user, "user", emotion_update_user_message) # Save this update as a user message
                
                history_for_slm = get_conversation_history(current_user, limit=10)
                # Pass the emotion update message as the user_message for get_gemma_response
                response_from_slm = get_gemma_response(emotion_update_user_message, user_prompts[current_user][current_emotion], history_for_slm)
                
                if response_from_slm.startswith("Error:"):
                    display_message(response_from_slm, "error_msg")
                else:
                    save_message(current_user, "model", response_from_slm)
                    display_message("Agente: " + response_from_slm, "agent_msg")
            else:
                display_message(f"Sigues pareciendo {current_emotion}. No hay cambio en mi tono de conversación.", "info_msg")
        else:
            display_message("No se pudo detectar una emoción en la nueva foto.", "error_msg")
            
    except Exception as e:
        messagebox.showerror("Error de Visión", f"Ocurrió un error al procesar la foto con el módulo de visión: {e}")
        display_message(f"Error al procesar la foto de visión para actualización: {e}", "error_msg")
    finally:
        progress_bar.stop()
        update_emotion_button.config(state=tk.NORMAL)
        update_emotion_photo_entry.config(state=tk.NORMAL)
        chat_window.update_idletasks()


def create_gui():
    global chat_window, chat_display, user_entry, send_button
    global initial_photo_path_entry, initial_photo_button, process_initial_button
    global update_emotion_photo_entry, update_emotion_button, progress_bar

    setup_database()

    chat_window = tk.Tk()
    chat_window.title("🤖 Agente de Chat con Visión (Gemma 2B Local)")
    chat_window.geometry("800x850") # Slightly larger window
    chat_window.resizable(False, False)
    
    # --- Modern Color Palette ---
    PRIMARY_COLOR = "#4A90E2"  # Blue for accents and buttons
    SECONDARY_COLOR = "#F7F9FC" # Light background for frames
    ACCENT_COLOR = "#6CC091"   # Green for agent messages
    USER_COLOR = "#5D5C61"     # Darker gray for user messages
    ERROR_COLOR = "#E74C3C"    # Red for errors
    INFO_COLOR = "#8E9DAA"     # Muted gray for info
    SYSTEM_COLOR = "#9B59B6"   # Purple for system messages
    WINDOW_BG = "#E8EDF3"     # Very light blue-gray for main window background

    chat_window.config(bg=WINDOW_BG)

    # --- Configure Styles for ttk widgets ---
    style = ttk.Style()
    style.theme_use('clam') # 'clam' provides a good base for customization

    # General frame and label styles
    style.configure('TFrame', background=SECONDARY_COLOR)
    style.configure('TLabelFrame', background=SECONDARY_COLOR, foreground=PRIMARY_COLOR,
                    font=('Segoe UI', 12, 'bold'), borderwidth=1, relief="flat")
    style.configure('TLabel', background=SECONDARY_COLOR, foreground="#333333", font=('Segoe UI', 10))

    # Button styles
    style.configure('TButton', font=('Segoe UI', 10, 'bold'), background=PRIMARY_COLOR, foreground='white',
                    relief='flat', padding=(10, 5))
    style.map('TButton',
              background=[('active', '#357ABD'), ('pressed', '#357ABD')],
              foreground=[('active', 'white'), ('pressed', 'white')])

    # Entry widget style
    style.configure('TEntry', font=('Segoe UI', 11), padding=5, fieldbackground='white')

    # ScrolledText (chat display) styles - requires tag_config
    # (these are applied directly to the widget, not via ttk style)

    # Progressbar style
    style.configure('TProgressbar', background=PRIMARY_COLOR, troughcolor=SECONDARY_COLOR, borderwidth=1, relief='groove')


    # --- Main Title ---
    title_label = ttk.Label(chat_window, text="🚀 Agente de Conversación Inteligente con Visión",
                            font=("Segoe UI", 18, "bold"), foreground=PRIMARY_COLOR, background=WINDOW_BG)
    title_label.pack(pady=20)

    # --- Frame for Initial Photo Load ---
    initial_photo_frame = ttk.LabelFrame(chat_window, text="📸 1. Identificación Inicial", padding=(15, 15))
    initial_photo_frame.pack(pady=10, padx=25, fill="x")

    tk.Label(initial_photo_frame, text="Ruta de la foto inicial:", font=('Segoe UI', 10)).grid(row=0, column=0, padx=5, pady=5, sticky="w")
    initial_photo_path_entry = ttk.Entry(initial_photo_frame, width=50)
    initial_photo_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    initial_photo_button = ttk.Button(initial_photo_frame, text="Buscar Foto...", command=select_initial_photo)
    initial_photo_button.grid(row=0, column=2, padx=5, pady=5)

    process_initial_button = ttk.Button(initial_photo_frame, text="Procesar Foto", command=process_initial_photo)
    process_initial_button.grid(row=0, column=3, padx=5, pady=5)
    initial_photo_frame.grid_columnconfigure(1, weight=1) # Allow entry to expand

    # --- Chat Display Area ---
    chat_display_frame = ttk.Frame(chat_window, relief="solid", borderwidth=1, padding=10, style='ChatDisplay.TFrame')
    chat_display_frame.pack(pady=10, padx=25, fill="both", expand=True)
    style.configure('ChatDisplay.TFrame', background='white', bordercolor='#DDDDDD') # Lighter border for chat

    chat_display = scrolledtext.ScrolledText(chat_display_frame, wrap=tk.WORD, state=tk.DISABLED,
                                             font=("Segoe UI", 11), bg="#ffffff", fg="#333333", relief="flat", padx=10, pady=10)
    chat_display.pack(fill="both", expand=True)

    # Configure tags for message colors (using the new palette)
    chat_display.tag_config("user_msg", foreground=USER_COLOR, font=("Segoe UI", 11, 'bold'))
    chat_display.tag_config("agent_msg", foreground=ACCENT_COLOR, font=("Segoe UI", 11))
    chat_display.tag_config("error_msg", foreground=ERROR_COLOR, font=("Segoe UI", 11, 'bold'))
    chat_display.tag_config("info_msg", foreground=INFO_COLOR, font=("Segoe UI", 10, 'italic'))
    chat_display.tag_config("system_msg", foreground=SYSTEM_COLOR, font=("Segoe UI", 11, 'bold'))


    # --- Input Message Frame ---
    input_frame = ttk.Frame(chat_window, padding=(10, 10))
    input_frame.pack(pady=5, padx=25, fill="x")

    user_entry = ttk.Entry(input_frame, font=("Segoe UI", 11))
    user_entry.pack(side=tk.LEFT, fill="x", expand=True, padx=5, ipady=3)
    user_entry.bind("<Return>", lambda event=None: send_message_gui())

    send_button = ttk.Button(input_frame, text="Enviar ✉️", command=send_message_gui)
    send_button.pack(side=tk.RIGHT, padx=5)

    # --- Progress Bar (New) ---
    progress_bar = ttk.Progressbar(chat_window, orient="horizontal", length=200, mode="indeterminate", style='TProgressbar')
    progress_bar.pack(pady=5, padx=25, fill="x")


    # --- Frame for Updating Emotion ---
    update_emotion_frame = ttk.LabelFrame(chat_window, text="🔄 2. Actualizar Emoción", padding=(15, 15))
    update_emotion_frame.pack(pady=10, padx=25, fill="x")

    tk.Label(update_emotion_frame, text="Nueva foto:", font=('Segoe UI', 10)).grid(row=0, column=0, padx=5, pady=5, sticky="w")
    update_emotion_photo_entry = ttk.Entry(update_emotion_frame, width=50)
    update_emotion_photo_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    update_emotion_button = ttk.Button(update_emotion_frame, text="Detectar Emoción", command=select_update_emotion_photo)
    update_emotion_button.grid(row=0, column=2, padx=5, pady=5)
    update_emotion_frame.grid_columnconfigure(1, weight=1) # Allow entry to expand

    # Initial state: disable chat and emotion update until user identified
    user_entry.config(state=tk.DISABLED)
    send_button.config(state=tk.DISABLED)
    update_emotion_photo_entry.config(state=tk.DISABLED)
    update_emotion_button.config(state=tk.DISABLED)

    display_message("¡Bienvenido al Agente de Conversación con Visión! 🎉", "system_msg")
    display_message("Para comenzar, por favor, carga una foto inicial para que pueda identificarte y entender tu emoción. Luego, haz clic en 'Procesar Foto'.", "info_msg")
    display_message(f"Asegúrate de que tu modelo '{GEMMA_MODEL_NAME}' está corriendo localmente (ej. con Ollama) en {GEMMA_API_URL}", "info_msg")

    chat_window.mainloop()

if __name__ == '__main__':
    create_gui()