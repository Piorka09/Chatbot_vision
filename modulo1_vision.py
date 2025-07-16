# 1) Importaciones y configuración
import os
import cv2
import logging
import time
import numpy as np
import pandas as pd
from mtcnn import MTCNN
from tqdm import tqdm
from tensorflow.keras import Sequential, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam



logging.basicConfig(level=logging.INFO)

personas = ['personaA', 'personaB']
emociones = [
    'alegre','triste','pensativo',
    'con_ira','cansado','sorprendido','riendo'
]

base_train = 'data/train'
base_test = 'data/test'

# 2) Detector MTCNN y función de recorte (Inicializa el modelo MTCNN que se usará para detectar caras en las imágenes.)
detector = MTCNN()

def detectar_y_recortar_cara(ruta_img,
                             max_size=2000,
                             max_time=3.0,
                             target_size=(224,224)):
    img = cv2.imread(ruta_img)
    if img is None:
        logging.warning(f"No se pudo cargar la imagen: {ruta_img}")
        return None
    # Si la imagen ya tiene el tamaño deseado, la devuelve directamente
    if img.shape[0:2] == target_size: return img
    # Si la imagen es demasiado grande, la ignora
    if img.shape[0] > max_size or img.shape[1] > max_size:
        logging.warning(f"Imagen demasiado grande para procesar: {ruta_img}")
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    start = time.time()
    resultados = detector.detect_faces(rgb)
    # Si no se detectan caras o la detección excede el tiempo máximo, ignora la imagen
    if not resultados or time.time()-start > max_time:
        logging.warning(f"No se detectaron caras o el tiempo de detección excedió el límite para: {ruta_img}")
        return None
    # Extrae las coordenadas de la primera cara detectada
    x,y,w,h = resultados[0]['box']
    # Asegura que las coordenadas no sean negativas
    x, y = max(0,x), max(0,y)
    # Recorta la región de interés (ROI)
    roi = img[y:y+h, x:x+w]
    # Redimensiona la ROI al tamaño objetivo
    return cv2.resize(roi, target_size)

# 3) Preprocesa in-place todas las imágenes con MTCNN
# Esta función itera a través de los directorios de entrenamiento (base_train) para cada persona y emoción.
def procesar_train():
    logging.info("Iniciando preprocesamiento de imágenes de entrenamiento...")
    for p in personas:
        for e in emociones:
            carpeta = os.path.join(base_train, p, e)
            if not os.path.isdir(carpeta):
                logging.warning(f"Directorio no encontrado: {carpeta}. Saltando.")
                continue
            # Itera sobre los archivos en la carpeta con una barra de progreso
            for f in tqdm(os.listdir(carpeta), desc=f"Procesando {p}/{e}"):
                # Verifica si el archivo es una imagen
                if not f.lower().endswith(('.jpg','jpeg','png')):
                    continue
                ruta = os.path.join(carpeta, f)
                rec = detectar_y_recortar_cara(ruta)
                if rec is not None:
                    try:
                        # Guarda la imagen recortada y redimensionada en la misma ruta, sobrescribiendo la original
                        cv2.imwrite(ruta, rec)
                    except Exception as err:
                        logging.error(f"Error al guardar la imagen procesada {ruta}: {err}")
                else:
                    # Opcionalmente, puedes mover o registrar imágenes que no pudieron ser procesadas
                    logging.info(f"No se pudo procesar la imagen (no se detectó cara o error): {ruta}")
    logging.info("Preprocesamiento de entrenamiento completado.")



# 4) Construye DataFrame con rutas y etiquetas
logging.info("Construyendo DataFrame con rutas y etiquetas...")
filepaths, labels = [], []
for p in personas:
    for e in emociones:
        carpeta = os.path.join(base_train, p, e)
        if not os.path.isdir(carpeta): continue
        for f in os.listdir(carpeta):
            if not f.lower().endswith(('.jpg','jpeg','png')):
                continue
            filepaths.append(os.path.join(carpeta, f))
            labels.append(f"{p}_{e}")

df = pd.DataFrame({'filename': filepaths, 'class': labels})
logging.info(f"DataFrame creado con {len(df)} entradas.")

# 5) Split estratificado en train/val 
logging.info("Realizando split estratificado de datos...")
df_train, df_val = train_test_split(
    df, test_size=0.2,
    stratify=df['class'],
    random_state=42
)
logging.info(f"Muestras de entrenamiento: {len(df_train)}, Muestras de validación: {len(df_val)}")

# 6) Generadores con augmentations moderadas
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normaliza los píxeles a un rango de 0 a 1
    rotation_range=30,         # Rota imágenes aleatoriamente hasta 30 grados
    width_shift_range=0.2,     # Desplaza imágenes horizontalmente
    height_shift_range=0.2,    # Desplaza imágenes verticalmente
    zoom_range=0.2,            # Aplica zoom aleatorio
    horizontal_flip=True,      # Voltea imágenes horizontalmente
    brightness_range=(0.8,1.2), # Ajusta el brillo aleatoriamente
    fill_mode='nearest'        # Rellena los puntos nuevos creados por las transformaciones
)
val_datagen = ImageDataGenerator(rescale=1./255) # Solo normaliza para validación

logging.info("Configurando generadores de imágenes...")
train_gen = train_datagen.flow_from_dataframe(
    df_train, x_col='filename', y_col='class',
    target_size=(224,224), batch_size=batch_size,
    class_mode='categorical', shuffle=True
)
val_gen = val_datagen.flow_from_dataframe(
    df_val, x_col='filename', y_col='class',
    target_size=(224,224), batch_size=batch_size,
    class_mode='categorical', shuffle=False
)

# 7) Cálculo de class_weight/ Calcula pesos para cada clase
logging.info("Calculando pesos de clase...")
weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weight_dict = dict(enumerate(weights))
logging.info(f"Pesos de clase calculados: {class_weight_dict}")

# ---- Definición de num_classes (numero de clases)  --
num_classes = df['class'].nunique()
logging.info(f"Número total de clases para el modelo: {num_classes}")

# 8) Definición de la CNN 
logging.info("Definiendo el modelo CNN personalizado (desde cero)...")

model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25), # Mayor dropout en las primeras capas si se entrena desde cero

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.GlobalAveragePooling2D(), # Reemplaza Flatten para mejor robustez
    layers.Dense(256, activation='relu'), # Capa densa más grande
    layers.BatchNormalization(),
    layers.Dropout(0.5), # Mayor dropout en las capas densas

    layers.Dense(num_classes, activation='softmax')
])

logging.info("Modelo CNN personalizado definido.")
model.summary()


# 9) Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10, # Aumenta la paciencia un poco, ya que podría tardar más en converger desde cero
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5, # Aumenta la paciencia para reducción de LR
    min_lr=1e-7 # Mínima tasa de aprendizaje
)
checkpoint = ModelCheckpoint(
    'mejor_modelo.h5',
    monitor='val_loss',
    save_best_only=True
)
logging.info("Callbacks configurados.")

# 10) Compilación con learning rate inicial
opt = Adam(learning_rate=1e-3)

model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
logging.info("Modelo compilado.")

# 11) Entrenamiento (aumentar épocas, ya que se entrena desde cero)
logging.info("Iniciando entrenamiento del modelo...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30, # Aumentar el número de épocas para entrenamiento desde cero
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weight_dict,
    verbose=2
)
logging.info("Entrenamiento completado.")

# 12) Mapeo índice → etiqueta
idx_to_label = {v: k for k, v in train_gen.class_indices.items()}
logging.info("Mapeo de índice a etiqueta creado.")

# Cargar el mejor modelo después del entrenamiento
model = None # Asegurarse de que el modelo se carga limpio
try:
    model = load_model('mejor_modelo.h5')
    logging.info("Mejor modelo cargado exitosamente.")
except Exception as e:
    logging.error(f"Error al cargar el mejor modelo: {e}. Asegúrate de que 'mejor_modelo.h5' exista y esté entrenado.")

# 13) Inferencia sobre nuevas imágenes (Retornando Persona, Emoción, y Probabilidades)
def predecir_emocion_y_persona(ruta_img):
    global model, idx_to_label # Declarar globales para asegurar acceso
    
    if model is None:
        logging.error("El modelo no está cargado. No se puede realizar la predicción.")
        return None, None, None # ¡Siempre retorna 3 valores!

    face = detectar_y_recortar_cara(ruta_img)
    if face is None:
        print(f"No se detectó cara o hubo un error de procesamiento en {ruta_img}")
        return None, None, None # ¡Siempre retorna 3 valores!

    x = face.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)

    if 'idx_to_label' not in globals() or idx_to_label is None:
        logging.error("El mapeo de índice a etiqueta (idx_to_label) no está disponible. No se puede interpretar la predicción.")
        return None, None, None # ¡Siempre retorna 3 valores!

    try:
        probs = model.predict(x, verbose=0)[0]
        
        # Validar que probs no esté vacío o sea un valor no numérico
        if not isinstance(probs, np.ndarray) or probs.size == 0 or np.isnan(probs).any():
            logging.error(f"La predicción de probabilidades es inválida para {ruta_img}.")
            return None, None, None

        idx_pred = np.argmax(probs)
        label_pred = idx_to_label[idx_pred]
        person_pred = label_pred.split('_')[0]
        emotion_pred = label_pred.split('_')[1]
        
        return person_pred, emotion_pred, probs # ¡Siempre retorna 3 valores!
    except Exception as e:
        logging.error(f"Error durante la predicción de {ruta_img}: {e}")
        return None, None, None # ¡Siempre retorna 3 valores en caso de error!

print("\n--- Inferencia en 'data/test' ---")
if os.path.isdir(base_test):
    test_images = [f for f in os.listdir(base_test) if f.lower().endswith(('.jpg','jpeg','png'))]
    if test_images:
        for f in tqdm(test_images, desc="Realizando inferencia"):
            ruta_completa = os.path.join(base_test, f)
            person_detected, emotion_detected, all_probs = predecir_emocion_y_persona(ruta_completa)
            
            if person_detected and emotion_detected and all_probs is not None:
                print(f"\nArchivo: {os.path.basename(f)}")
                print(f"Predicción Principal: Persona: {person_detected}, Emoción: {emotion_detected}")
                
                # Asegurarse de que hay suficientes elementos para el top 3
                if len(all_probs) >= 3:
                    top_3_indices = np.argsort(all_probs)[::-1][:3]
                    print("Top 3 Predicciones (Clase_Completa - Probabilidad):")
                    for i in top_3_indices:
                        predicted_label = idx_to_label[i]
                        probability = all_probs[i]
                        print(f"   - {predicted_label}: {probability:.4f}")
                else:
                    print("Menos de 3 clases disponibles para mostrar el top.")
                    for i, prob in enumerate(all_probs):
                        predicted_label = idx_to_label[i]
                        print(f"   - {predicted_label}: {prob:.4f}")

            else:
                print(f"No se pudo procesar la imagen {os.path.basename(f)} para inferencia.")
    else:
        print(f"No se encontraron imágenes en '{base_test}'.")
else:
    print(f"Directorio de prueba '{base_test}' no encontrado. No se realizará inferencia.")