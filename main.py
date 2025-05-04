import os
import sys
import shutil
import json as js
import cv2 as cv
from ultralytics import YOLO

def loadModel():
    trainNumber = 1

    if os.path.exists("./runs/classify"):
        while os.path.exists(f"./runs/classify/train{trainNumber}"):
            trainNumber += 1
            continue
    
        if trainNumber == 1:
            return YOLO(f"./runs/classify/train/weights/best.pt")
        else:
            return YOLO(f"./runs/classify/train{trainNumber-1}/weights/best.pt")
    else:
        print("No model were found under directory './runs/classify', train one with ultralytics YOLO")
        sys.exit(1)

def loadJson():
    try:
        with open("./paintingdb.json") as js_file:
            return js.load(js_file)
    except:
        print("File './paintingdb.json' not found, re-download the repository or download the single file")
        sys.exit(1)

def main():
    model = loadModel()
    painting_data = loadJson()

    if os.path.exists("./temp"):
        shutil.rmtree("./temp")
    os.mkdir("./temp")


    while True:

        cam_port = 0
        cam = cv.VideoCapture(cam_port)
        cam.set(3, 640)
        cam.set(4, 640)

        # Loop for getting an image
        while True:
            ret, frame = cam.read()
            cv.imshow('frame', frame)

            pressedKey = cv.waitKey(1) & 0xFF

            if pressedKey == ord('s'):
                cv.imwrite("./temp/frame.jpeg", frame)
                break
            elif pressedKey == ord('q'):
                shutil.rmtree("./temp")
                sys.exit(0)
        
        cam.release()
        cv.destroyAllWindows()

        model("./temp/frame.jpeg")
        
        
main()