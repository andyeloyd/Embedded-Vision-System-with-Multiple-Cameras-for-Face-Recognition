import cv2 as cv
from threading import Thread
import time
import tensorflow as tf

import pandas as pd
import datetime

from face_detection import mtcnn_detector
from image_preprocessing import *
from face_recognition import *



#src_1 = "rtsp://admin:vision1.@192.168.1.65:554/cam/realmonitor?channel=0&subtype=00"
#src_2 = "rtsp://admin:vision1.@192.168.1.102:554/cam/realmonitor?channel=0&subtype=00"


# Limitar alojamiento de GPU para TensorFlow

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


# Camera divida en hilos para procesamiento mas rapido
# mas informacion en: 
# https://stackoverflow.com/questions/58293187/opencv-real-time-streaming-video-capture-is-slow-how-to-drop-frames-or-get-sync
class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv.VideoCapture(src)
        self.capture.set(cv.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)

    def show_frame(self):
        cv.imshow('frame', self.frame)
        cv.waitKey(self.FPS_MS)
    def get_frame(self):
        return self.frame


def authentication_logging(ids_found, ids_auth, ids_count, ids_patience, auth_threshold, pat_threshold,
                           update_ticks, update_register, update_interval=100):
    # Si ID ha sido encontrada, es agregada a lista de autenticacion
    for idx in range(len(ids_found)):
        if ids_found[idx]:
            ids_auth[idx] = True

    # Revisa el estado de cada ID en lista de autenticacion.
    for idx in range(len(ids_auth)):

        if ids_auth[idx]:
            # If ID in authentication has been found again, add 1 to count.
            # Si ID en lista de autenticacion ha sido encontrada de nuevo, se agrega 1 a la cuenta de la ID.
            if ids_found[idx]:
                ids_count[idx] += 1
                # Si la cuenta de la ID ha rebasado el umbral, se hace registro del avistamiento.
                if ids_count[idx] > auth_threshold:
                    dt_obj = datetime.datetime.now()
                    df.iat[idx, 1] = dt_obj.date()
                    df.iat[idx, 2] = dt_obj.time()
                    update_register = True
                    print('ID %d has been authenticated. Time: %s' % (idx, dt_obj))
                    # Retirada de ID de lista de autenticacion, reinicio de cuenta y paciencia.
                    ids_auth[idx] = False
                    ids_count[idx] = 0
                    ids_patience[idx] = 0
                # Retirada de ID de lista de IDs encontradas
                ids_found[idx] = False
            # Si ID en lista de autenticacion NO ha sido encontrada de nuevo, se agrega 1 a la paciencia de ID.
            else:
                ids_patience[idx] += 1
                # Si la paciencia de la ID ha rebasado el umbral, se hace retira la ID de la lista de autenticacion
                # reinicio de cuenta y paciencia.
                if ids_patience[idx] > pat_threshold:
                    ids_auth[idx] = False
                    ids_count[idx] = 0
                    ids_patience[idx] = 0


    # Actualiza el archivo CSV cada cierto numero de cuadros si se han realizado avistamientos
    update_ticks += 1
    if update_register and update_ticks >= update_interval:

      df.to_csv(df_csv, index=False)
      print('CSV file updated.')
      update_register = False
      update_ticks = 0
    return ids_found, ids_auth, ids_count, ids_patience, update_ticks, update_register

'''
def draw_bounding_box(img, bbox):
    startY, startX, endY, endX = bbox
    cv.rectangle(img, (startX, startY), (endX, endY),(255, 0, 0), 2)
    return img

def write_prediction_data(img, bbox, name, probability):
    startY, startX, endY, endX = bbox
    text = "{}: {:.2f}%".format(name, probability)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv.rectangle(img, (startX, startY), (endX, endY),
                  (0, 0, 255), 2)
    cv.putText(img, text, (startX, y),
                cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return img
'''

# Inicializacion de camara(s)
# Si se quieren agregar mas camaras, basta con agregarlas a la lista de camaras.

camera_list = []
src_1 = "rtsp://admin:vision1.@192.168.1.65:554/cam/realmonitor?channel=0&subtype=00"
threaded_camera_1 = ThreadedCamera(src_1)
camera_list.append(threaded_camera_1)
#src_2 = "rtsp://admin:vision1.@192.168.1.102:554/cam/realmonitor?channel=0&subtype=00"
#threaded_camera_2 = ThreadedCamera(src_2)
#camera_list.append(threaded_camera_2)

# Inicializacion de red de y parametros para deteccion facial
MTCNN = mtcnn_detector()
min_size = 50
factor = 0.7
#thresholds = [0.6, 0.7, 0.8]
thresholds = [0.6, 0.75, 0.9]

# Inicializacion de red de reconocimiento facial
trt_engine_path = '/home/nvidia/Documents/Face_recognition/resnet/Resnet50_new_imp/tensorrt_engine/resnet50_engine.trt'
fr_model = TrtModel(trt_engine_path)
probability_threshold = 93

# Diccionario de IDs
classDict = {0:'Andres',1:'Laura', 2:'Abue', 3:'Gera',
             4:'14th Dalai Lama', 5:'Ayami', 6:'Abdullah II',
             7:'Aco Petrovic', 8:'Adhyayan Suman', 9:'Aditya Seal',
             10:'Agata Passent', 11:'Ahmet Davutoglu', 12:'Airi Suzuki',
             13:'Aishwarya Rai', 14:'Alain Traore', 15:'Alex Gonzaga',
             16:'Alexandra Edenborough', 17:'Alodia Gosiengfiao', 18:'Amber Brkich',
             19:'Amina Shafaat', 20:'Ana Rosa Quintana', 21:'Andrea Anders',
             22:'Andrew Upton', 23:'Angelique Kidjo'}


# Determinacion de umbrales para registro
# auth_threshold: cantidad de veces que se debe de ver a la ID para registrarla.
# pat_threshold: numerode cuadros en los que se permite no encontrar la ID antes de descartarla.
n_class = 24
auth_threshold = 8
pat_threshold = 4

# Abrir CSV de registro de entradas
df_csv = "./login_data.csv"
df = pd.read_csv(df_csv, skipinitialspace=True)
# Cada cuantos cuadros actualizar archivo CSV.
csv_update_rate = 100

# Inicializacion de variables de control.
ids_found = [False] * n_class
ids_auth = [False] * n_class
ids_count = [0] * n_class
ids_patience = [0] * n_class
update_ticks = 0
update_register = False




# Funcionamiento con varias camaras

while True:
    for threaded_camera in camera_list:

        try:
            img = threaded_camera.get_frame()
            detections = MTCNN(img, min_size, factor, thresholds)
            if len(detections[0]) > 0:
                # Ciclo en cada caja delimitadora obtenida
                for bbox in detections[0]:
                    bbox = [int(i) for i in bbox]


                    # Reconocimiento de rostros
                    cropped_face = crop_face(img, bbox)
                    cropped_face = cv.resize(cropped_face, (224,224))
                    cropped_face = preprocess_input_image(cropped_face)
                    result = fr_model(cropped_face,1)
                    result = np.array(result).squeeze()
                    prediction = int(np.argmax(result, axis=0))
                    probability = 100 * (np.amax(result, axis=0))


                    # Si la certidumbre de la prediccion esta por encime del umbral,
                    # el avistamiento sera procesado por el control de registros de ingreso.
                    if probability >= probability_threshold:
                        ids_found[prediction] = True


            # Control de registros de ingreso
            ids_found, ids_auth, ids_count, ids_patience, update_ticks, update_register =\
                authentication_logging(ids_found, ids_auth, ids_count, ids_patience, auth_threshold, pat_threshold,
                                       update_ticks, update_register, csv_update_rate)

        except AttributeError:
            pass


