import cv2 as cv
from threading import Thread
import time
import tensorflow as tf

from face_detection import mtcnn_detector
from image_preprocessing import *
from face_recognition import *


# Camaras en la red
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


# Inicializacion de camara
src = "rtsp://admin:vision1.@192.168.1.65:554/cam/realmonitor?channel=0&subtype=00"
threaded_camera = ThreadedCamera(src)

# Inicializacion de red de y parametros para deteccion facial
MTCNN = mtcnn_detector()
min_size = 50
factor = 0.7
#thresholds = [0.6, 0.7, 0.8]
thresholds = [0.6, 0.75, 0.9]

# Inicializacion de red de reconocimiento facial
trt_engine_path = '/home/nvidia/Documents/Face_recognition/resnet/Resnet50_new_imp/tensorrt_engine/resnet50_engine.trt'
fr_model = TrtModel(trt_engine_path)
probability_threshold = 90

# Diccionario de IDs
classDict = {0:'Andres',1:'Laura', 2:'Abue', 3:'Gera',
             4:'14th Dalai Lama', 5:'Ayami', 6:'Abdullah II',
             7:'Aco Petrovic', 8:'Adhyayan Suman', 9:'Aditya Seal',
             10:'Agata Passent', 11:'Ahmet Davutoglu', 12:'Airi Suzuki',
             13:'Aishwarya Rai', 14:'Alain Traore', 15:'Alex Gonzaga',
             16:'Alexandra Edenborough', 17:'Alodia Gosiengfiao', 18:'Amber Brkich',
             19:'Amina Shafaat', 20:'Ana Rosa Quintana', 21:'Andrea Anders',
             22:'Andrew Upton', 23:'Angelique Kidjo'}

while True:
    try:
        #threaded_camera.show_frame()
        img = threaded_camera.get_frame()
        #cv.imshow('frame', img)
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

                # Dibujo de caja delimitadora
                img = draw_bounding_box(img, bbox)
                # Si se reconocio una identidad por encima del umbral, escribe informacion en imagen
                if probability >= probability_threshold:
                    img = write_prediction_data(img, bbox, classDict[prediction], probability)


        cv.imshow('frame', img)


        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    except AttributeError:
        pass

