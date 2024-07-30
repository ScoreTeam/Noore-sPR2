# common dependencies
import glob
import os,cv2,warnings,logging,sys
from typing import Any, Dict, List, Union, Optional
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import pandas as pd
import tensorflow as tf
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_path)
from deepface.commons import package_utils
from deepface.commons import logger as log
from deepface.modules import (
    representation,
    recognition,
    detection,
)

logger = log.get_singletonish_logger()
package_utils.validate_for_keras3()

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf_version = package_utils.get_tf_major_version()
tf.get_logger().setLevel(logging.ERROR)
def FindFaceFromImage(image_path,silent):
    frame = cv2.imread(image_path)

    result = find(img_path=image_path, db_path='DatabaseFR', enforce_detection=False, model_name='VGGFace',silent=silent)
    # if silent == False:print(result)
    print(result)
    best_index = 0
    distanceThreshold=0.03

    

    for res in result:
        
        if 'identity' in res and len(res['identity']) > 0 and res['distance'][best_index]<distanceThreshold: 

            print("in here")
            name = res['identity'][best_index].split("\\")[1].split(".")[0]
            xmin = int(res['source_x'][best_index])
            ymin = int(res['source_y'][best_index])
            w = res['source_w'][best_index]
            h = res['source_h'][best_index]
            xmax = int(xmin + w)
            ymax = int(ymin + h)
            dis=res['distance'][best_index]
            # id = name[-1]
            id = name.split('_')[-1]

            # Draw rectangle and put text on the frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (1, 255, 1), 1)
            cv2.putText(frame, name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            newimagepath="RecFaces/Detectedface.jpg"
            increameant=1
            while os.path.exists(newimagepath):
                newimagepath="RecFaces/Detectedface_"+str(increameant)+".jpg"
                increameant+=1
            cv2.imwrite(newimagepath,frame)
            # modify the id thing to cal after _
            if name[:8]=="Customer":
                return ["customer",id,name,xmin,ymin,w,h,xmax,ymax,dis]
            else:   
                return ["employee",id,name,xmin,ymin,w,h,xmax,ymax,dis]
        else:
            # im adding the new customer faces into the database
            print("adding new customer...")
            increameant=200
            newimagepath=f"DatabaseFR/Customer_{increameant}.jpg"
            while os.path.exists(newimagepath):
                increameant+=1
                newimagepath="DatabaseFR/Customer_"+str(increameant)+".jpg"
            # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (1, 255, 1), 1)
            # cv2.putText(frame, "customer", (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imwrite(newimagepath,frame)
            """"
            note: discuss this (im currently send the customer with no identity in the database as empty, so he will skip a frame but in the next frame he will back claed)
            """
        return ["customer",increameant,f"Customer{increameant}",0,0,0,0,0,0,0,]
            
            
def find(
    img_path: Union[str, np.ndarray],
    db_path: str,
    model_name: str = "ArcFace",
    distance_metric: str = "cosine", # this can be euclidean (we will test it later)
    enforce_detection: bool = True,
    detector_backend: str = "opencv", #this can be :'opencv', 'retinaface','mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip' (We will test all them late)
    align: bool = True, # alignment based on the eye positions
    expand_percentage: int = 0, #expand detected facial area with a percentage
    threshold: Optional[float] = None, # we will test many threshold based on the work enviroment(the store we will be testing on)
    normalization: str = "base", #Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace
    silent: bool = False, #log messages for a quieter analysis
    refresh_database: bool = True, #Synchronizes the images representation (pkl) file with the directory/db files,f set to false, it will ignore any file changes inside the db_path
    anti_spoofing: bool = False, #for fake images 
) -> List[pd.DataFrame]:
    
    return recognition.find(
        img_path=img_path,
        db_path=db_path,
        model_name=model_name,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        threshold=threshold,
        normalization=normalization,
        silent=silent,
        refresh_database=refresh_database,
        anti_spoofing=anti_spoofing,
    )


def represent(
    img_path: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    anti_spoofing: bool = False,
) -> List[Dict[str, Any]]:
    return representation.represent(
        img_path=img_path,
        model_name=model_name,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        normalization=normalization,
        anti_spoofing=anti_spoofing,
    )

def extract_faces(
    img_path: Union[str, np.ndarray],
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    grayscale: bool = False,
    anti_spoofing: bool = False,
) -> List[Dict[str, Any]]:
    return detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        grayscale=grayscale,
        anti_spoofing=anti_spoofing,
    )

