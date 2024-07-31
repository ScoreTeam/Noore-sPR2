
'''
sequence:
* Yolo --> an image with n_bounding_boxes 
* FR: 
1) crop the full image into n images (each image is a bounding box)
2) FR each new image and sort them to customer or an employee
3) save the bounding box with the info (Detected boxes/labeled boxes)
* CalDest ..?


'''

import glob
from FaceRec.NFR2 import NFR
from PIL import Image
import os,cv2
from concurrent.futures import ThreadPoolExecutor, as_completed


def crop_images(image_path, bounding_boxes):
    cropped_images = []
    # to clear all of the folder past contents (we dont need the previous croped images right?)
    files = glob.glob(os.path.join("Croppedimages", '*'))
    for f in files:
        os.remove(f)
    
    with Image.open(image_path) as img:
        for box in bounding_boxes:
            # x, y, w, h = box
            x, y, w, h = box['x'], box['y'], box['w'], box['h']  
            # print(x, y, w, h)  
            crop_box = (x, y, x + w, y + h)
            cropped_img = img.crop(crop_box)
            # saving the cropped images in a folder called Croppedimages
            filename = f"cropped_image.jpg"
            new_image_path = os.path.join("Croppedimages/", filename)
            increment = 1
            while os.path.exists(new_image_path):
                new_image_path = os.path.join("Croppedimages/", f"cropped_image_{increment}.jpg")
                increment += 1
            cropped_img.save(new_image_path)
            # cropped_images.append(new_image_path)
            cropped_images.append((cropped_img,(x,y,w,h),new_image_path))
            
            
            
    return cropped_images
def FaceRec(new_images2, showlog):
    updated_images = []
    
    for i, (frame,coords,image_path) in enumerate(new_images2):
        if showlog:
            print("\n----------------------------------------------\n")
            print(f"Image {image_path} : \n")
            newfindings = NFR.FindFaceFromImage(image_path, silent=showlog)
            print(f"new finding:\n{newfindings}\n")
            updated_images.append((image_path, coords, newfindings,))
            print("\n----------------------------------------------\n")
        else :
            newfindings = NFR.FindFaceFromImage(image_path, silent=showlog)
            updated_images.append((image_path, coords, newfindings))
            
    
    return updated_images
# def process_image(frame, coords, image_path, showlog):
#     if showlog:
#         print("\n----------------------------------------------\n")
#         print(f"Image {image_path} : \n")
#     newfindings = NFR.FindFaceFromImage(image_path, silent=showlog)
#     if showlog:
#         print(f"new finding:\n{newfindings}\n")
#         print("\n----------------------------------------------\n")
#     return (image_path, coords, newfindings)

# def FaceRec(new_images2, showlog):
#     updated_images = []

#     with ThreadPoolExecutor() as executor:
#         futures = [
#             executor.submit(process_image, frame, coords, image_path, showlog)
#             for frame, coords, image_path in new_images2
#         ]
#         for future in as_completed(futures):
#             updated_images.append(future.result())
    
#     return updated_images

def get_image_heights(image_paths):
    heights = []
    for path in image_paths:
        with Image.open(path) as img:
            heights.append(img.height)
    return heights
def get_image_paths_from_folder(folder_path):
    image_paths = []
    image_paths.extend(glob.glob(os.path.join(folder_path, '*.jpg')))
    
    return image_paths
def ExtractFaces(images,showlog):
    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Create a directory to save extracted faces
    output_dir = 'extracted_faces'
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join("extracted_faces", '*'))
    for f in files:
        os.remove(f)
    for i, (frame,coords,image_path) in enumerate(images):
        # print(image_path)
    # Read the image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # Loop through detected faces and save each face
        for count, (x, y, w, h) in enumerate(faces):
            # Draw rectangle around the face
            # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Extract the face
            face = img[y:y+h, x:x+w]

            # Save the extracted face
            face_filename = os.path.join(output_dir, f'face_{count+1}.jpg')
            increment = 1
            while os.path.exists(face_filename):
                face_filename = os.path.join(f"{output_dir}/", f"face_{increment}.jpg")
                increment += 1
            cv2.imwrite(face_filename, face)
            # face.save(new_image_path)
            print(f"Face {count+1} saved as {face_filename}")
            

    
    

    


def DetectFace(image_path,bounding_boxes):
    
    # first i croped the images into a tuble (image,boundingbox)
    cropped_imgs = crop_images(image_path,bounding_boxes)
    # sec i will call the face rec for each image and return to you the labeled boxes
    # extracting faces for testing
    ExtractFaces(cropped_imgs,showlog=True)
    files = glob.glob(os.path.join("RecFaces", '*'))
    for f in files:
        os.remove(f)
    images=get_image_paths_from_folder("extracted_faces")
    heights=get_image_heights(images)
    # print(heights)
    # labeled_boxes=FaceRec(images,showlog=True)
    labeled_boxes=FaceRec(cropped_imgs,showlog=True)
   
    print("labeled_boxes: ",labeled_boxes)
    return labeled_boxes,heights

    

    
    
