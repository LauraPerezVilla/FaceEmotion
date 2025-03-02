# Import de librerias
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from windows_toasts import WindowsToaster, Toast

ALERT_EXIST = False

# Clases de emociones
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
num_classes = len(classes)

# Colores para las barras
colors = ['r', 'g', 'b', 'y', 'k', 'm', 'c']

# Configuración del gráfico de barras
plt.ion()
fig, ax = plt.subplots()
bars = ax.bar(classes, [0] * num_classes, color=colors)
ax.set_ylim(0, 1)
ax.set_ylabel("Probabilidad")
ax.set_title("Distribución de Emociones")

# Cargamos el modelo de detección de rostros
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Cargamos el modelo de detección de emociones
emotionModel = load_model("modelFEC.h5")

# Captura de video
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Función para predecir emociones
def predict_emotion(frame, faceNet, emotionModel):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces, locs, preds = [], [], []
    
    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")
            Xi, Yi = max(0, Xi), max(0, Yi)
            face = frame[Yi:Yf, Xi:Xf]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            faces.append(face)
            locs.append((Xi, Yi, Xf, Yf))
            preds.append(emotionModel.predict(face)[0])
    
    return (locs, preds)


def check_alert(self):
    global ALERT_EXIST

    ALERT_EXIST = False

def send_alert(prob):

    global ALERT_EXIST

    if prob > 0.4 and not ALERT_EXIST:
        toaster = WindowsToaster('Alerta de estrés')
        # Initialise the toast
        newToast = Toast()
        # Set the body of the notification
        newToast.text_fields = ['La persona tiene un nivel de estrés elevado']
        
        # cuando la alerta desaparece, permitir que se vuelva a crear
        newToast.on_dismissed = check_alert

        # And display it!
        toaster.show_toast(newToast)

        ALERT_EXIST = True

# Variables para calcular FPS
time_prevframe = time.time()

while True:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=640)
    locs, preds = predict_emotion(frame, faceNet, emotionModel)
    
    for (box, pred) in zip(locs, preds):
        (Xi, Yi, Xf, Yf) = box

        (angry,disgust,fear,happy,neutral,sad,surprise) = pred

        print("Angry: {} \n Disgust: {} \n Fear: {} \n Happy: {} \n Neutral: {} \n Sad: {} \n Surprise: {}".format(angry, disgust, fear, happy, neutral, sad, surprise))

        prod_bademotions = (angry + sad) / 2

        print(prod_bademotions)

        label = "{}: {:.0f}%".format(classes[np.argmax(pred)], max(pred) * 100)
        
        cv2.rectangle(frame, (Xi, Yi-40), (Xf, Yi), (255, 0, 0), -1)
        cv2.putText(frame, label, (Xi+5, Yi-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255, 0, 0), 3)

        # Actualiza las alturas del gráfico de barras
        for bar, height in zip(bars, pred):
            bar.set_height(height)
        fig.canvas.flush_events()

        send_alert(prod_bademotions)

    # Cálculo de FPS
    time_actualframe = time.time()
    fps = 1 / (time_actualframe - time_prevframe)
    time_prevframe = time_actualframe

    cv2.putText(frame, f"{int(fps)} FPS", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
