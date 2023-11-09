#coded by student
import numpy as np
import cv2
import dlib  # for face and landmark detection
from imutils import face_utils
from scipy.spatial import distance as dist
import onnxruntime as rt

#face recognition module
recog_model_path='face_recognition/python_scripts/face-recog.onnx'
providers = ['CPUExecutionProvider']
face_model = rt.InferenceSession(recog_model_path, providers=providers)

#load saved embeddings and their respective names(encoding database)
loaded_name = np.load('face_recognition/python_scripts/face_enc/names.npz')
names_array = loaded_name['names']

loaded_enc = np.load('face_recognition/python_scripts/face_enc/encodings.npz')
encoding = loaded_enc['names']

#the preprocess function used in inception(so we need it here)
#each input image should go through this function
def preprocess_input(x):
    x = x.astype('float32')
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
#distance functoin used to measure distance between embeddings
def distance(tensor1,tensor2):
    distance = np.sum(np.square(tensor1-tensor2), axis=-1)
    return distance
#function to output the closest embedding and its name to the query embedding
def find_name(face_enc):
    distances = [distance(face_enc, other_embedding) for other_embedding in encoding]
    min_index = np.argmin(distances)
    if distances[min_index]>0.6:
        return "unknown"
    name = names_array[min_index]
    return name


#anti-spoof model
model_path='face_recognition/python_scripts/anti-spoof.onnx'
input_layer='actual_input_1'
output_layer=['output1']
providers = ['CPUExecutionProvider']
m = rt.InferenceSession(model_path, providers=providers)


#Blink detection module
# defining a function to calculate the EAR
def calculate_EAR(eye):
    # calculate the vertical distances
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])
    # calculate the horizontal distance
    x1 = dist.euclidean(eye[0], eye[3])
  
    # calculate the EAR
    EAR = (y1+y2) / x1
    return EAR
# Variables for Blink detection
blink_thresh = 0.4
succ_frame = 2
count_frame = 0
# Eye landmarks
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
  

#Face detection module
detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor(
   'face_recognition/python_scripts/shape_predictor_68_face_landmarks.dat')
detect_face = dlib.get_frontal_face_detector()




def predict_fake(rio):
    input_image = cv2.resize(rio, (128, 128))
    input_image=np.expand_dims(input_image, axis=0)
    input_image=input_image.astype(np.float32)
    input_image = np.transpose(input_image, (0, 3, 1, 2))
    onnx_pred = m.run(output_layer, {input_layer:input_image })
    prob=onnx_pred[0][0][0]
    return prob



# # Open the camera

cap = cv2.VideoCapture(0)

width = 320
height = 240
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)


#number of continuous frame taken to decide,keep one for multiuser(not effective)
#Used to get the recent frames to decide if the input is input is spoof or not
#added to handle unstable output by anti-spoof classifier
sample_number = 50
count = 0
measures = np.zeros(sample_number, dtype=np.float)



while True:
    ret, img_bgr = cap.read()
    if ret is False:
        print ("Error grabbing frame from camera")
        break

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    #detect face
    faces = detect_face(img_gray)

    measures[count%sample_number]=0

    point = (0,0)
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y

        #get region where the face is
        roi = img_bgr[y:y+h, x:x+w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        prob=predict_fake(roi)
        measures[count % sample_number] = prob
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
        point = (x, y-5)


        shape = landmark_predict(img_gray, face)
        shape = face_utils.shape_to_np(shape)
        lefteye = shape[L_start: L_end]
        righteye = shape[R_start:R_end]
        left_EAR = calculate_EAR(lefteye)
        right_EAR = calculate_EAR(righteye)
        # Avg of left and right eye EAR
        avg = (left_EAR+right_EAR)/2

        if avg < blink_thresh:
            #when eyes are closed
                count_frame += 1  # incrementing the frame count
        else:
            #if eyes are not closed check if they were closed using frame count variable
            if ((count_frame >= succ_frame) & (0 not in measures) & (np.mean(measures) <= 0.3)):
                #if eyes are closed for more than two frame(found optimal while expermenting) and
                #if the first n predictions are obtained from anti spoof classifier and
                #if the mean of recent n predictions from the anti-spoof classifier 
                # are less than threshold(0.3 found optimal while expermenting)
                    roi=cv2.resize(roi, (128,128), interpolation = cv2.INTER_AREA)
                    roi=np.expand_dims(roi, axis=0)
                    roi=preprocess_input(roi)
                    enc1=face_model.run(['lambda_2'], {"input": roi})
                    face_encoding=enc1[0]
                    names=find_name(face_encoding)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img_bgr, 'Blink Detected', (30, 30),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                    cv2.putText(img=img_bgr, text=str(names), org=point, fontFace=font, fontScale=0.9,
                                color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                    #set count_frame = 0 for detecting new users who are not real (for blink detection)
                    if count_frame>=10:
                        count_frame=0
                    
            else:
                # else assume as imposter
                text = "Imposter Detected"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=img_bgr, text=text, org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255),
                            thickness=2, lineType=cv2.LINE_AA)
                count_frame = 0
        
                

    count+=1
    cv2.imshow('img_rgb', img_bgr)

    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        print (measures, np.mean(measures))
        break

cap.release()
cv2.destroyAllWindows()
