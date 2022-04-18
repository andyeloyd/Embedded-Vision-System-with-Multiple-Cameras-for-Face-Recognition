# Embedded-Vision-System-with-Multiple-Cameras-for-Face-Recognition
Prototype of a Deep Neural Network-based vision system for face recognition on a NVIDIA Jetson TX2 embedded device connected to a IP camera system.

The code implements a TensorRT model for performing face recognition on a given private dataset. The architecture followed is ResNet50, which was pre-trained on the VGGFace2 dataset and through transfer learning, adapted to recognize faces on a small private dataset + a few IDs contained in the VGGFace2 test split. The model was originally trained on TensorFlow and Keras, but later converted to TensorRT for deployment on the Jetson TX2 module.

The system consists of the following models in cascade:

   -MTCNN face detection model
   -ResNet50 face recognition model
  
These models are contained within face-detection.py and face_recognition.py, respectively.
Since this system was developed with a passive use in mind, most configurations are preset within the scripts, but a config file can be
added with ease.
By default, the code is preset to run on two separate IP cameras.
(However, a config file can easily be added)

The system is implemented in two different scripts, of which either can be run from the command line: 
    -face_recognition_tracking.py: The input for both cameras are displayed on screen. Faces detected above the detection threshold on those inputs are delimited by a bounding box. If those faces are recognized to belong to a given ID above a preset confidence value, the corresponding ID name will be written down below its box. Intended as a demo.

   -face_recognition_register.py: No windows are displayed, and the script continues to run without visual outputs.Sightings of the IDs on the database  are recorded in login_data.csv, which registers the date and time of the last sighting for each ID.
  
   
Working of face_recognition_register.py (in spanish) 
![image](https://user-images.githubusercontent.com/54008991/163889891-5b55080d-3755-4286-bd82-333063c4067a.png)
