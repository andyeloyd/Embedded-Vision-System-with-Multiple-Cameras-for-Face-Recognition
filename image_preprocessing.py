import numpy as np

def crop_face(img, bbox_coordinates):
    # If any value is negative, it becomes zero instead
    bbox_coordinates = remove_negatives(bbox_coordinates)

    y1, x1, y2, x2 = bbox_coordinates
    # Perform the cropping on the input image
    img = img[y1:y2, x1:x2]
    return img

def remove_negatives(bbox_coordinates):
    for i in range(len(bbox_coordinates)):
        if bbox_coordinates[i] < 0:
            bbox_coordinates[i] = 0
    return bbox_coordinates

def preprocess_input_image(img):
    # Preprocesa imagen de entrada para inferencia en modelo
    # Resta el promedio de la imagen de vggface2 a la imagen de entrada,
    # reduce el rango de 0-255 a 0-1, y agrega una dimension al inicio.

    #preprocessing_mean = (91.4953, 103.8827, 131.0912)
    #img = img[:, :, ::-1] - preprocessing_mean
    img = img[:, :, ::-1] - (91.4953, 103.8827, 131.0912)
    img= img / 255.0
    img= np.reshape(img, newshape=(1, 224, 224, 3)).astype(np.float32)
    return img
