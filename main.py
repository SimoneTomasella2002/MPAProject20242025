import os
import sys
import shutil
import keyboard
import pygame
import json as js
import cv2 as cv
from pynput.keyboard import Listener
from gtts import gTTS
from ollama import chat, ChatResponse
from ultralytics import YOLO

# Search for and load the latest model generated based on the number
def loadModel():
    trainNumber = 2

    if os.path.exists("./runs/classify"):
        while os.path.exists(f"./runs/classify/train{trainNumber}"):
            trainNumber = trainNumber + 1

        if trainNumber == 1:
            return YOLO(f"./runs/classify/train/weights/best.pt")
        else:
            return YOLO(f"./runs/classify/train{trainNumber-1}/weights/best.pt")
    else:
        print("No model were found under directory './runs/classify', train one with ultralytics YOLO")
        sys.exit(1)

# Transform the JSON into an object so it can be iterated over
def loadJson():
    try:
        with open("./paintingdb.json") as js_file:
            return js.load(js_file)
    except:
        print("File './paintingdb.json' not found, re-download the repository or download the single file")
        sys.exit(1)

# Generate and play audio
def loadTTS():
    with open("./temp/tmp.txt") as txtFile:
        content = txtFile.read()
                
        index = content.find("</think>")

        if (index != -1):
            content = content[index + len("</think>"):].lstrip()

    print("\n\n-- Generating audio --")

    ttsObj = gTTS(text=content, lang="en", slow=False)

    ttsObj.save("./temp/tmp.mp3")

    print("\n\n-- Audio ready, press x to stop --")

    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load("./temp/tmp.mp3")
    pygame.mixer.music.play()

    if (sys.platform == "win32"):
        while pygame.mixer.music.get_busy():
            if keyboard.is_pressed('x'):
                pygame.mixer.music.stop()
                break
            pygame.time.Clock().tick(10)
    else:
        print("\n\n-- Can't find a better way to achieve the same result on linux, probably going to be addressed in the following revisions --")

        pygame.display.set_mode((1, 1))

        while pygame.mixer.music.get_busy():
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_x:
                        pygame.mixer.music.stop()
                        break
            pygame.time.Clock().tick(10)

    pygame.quit()

def main():
    llm_model = "deepseek-r1:1.5b"
    class_model = loadModel()
    paintings = loadJson()["paintings"]

    # temp contains the captured image and the generated audio
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

        result = class_model("./temp/frame.jpeg")

        # The class name is taken from the overall probabilities
        probs_list = result[0].probs.data.tolist()
        max_prob = max(probs_list)
        max_index = probs_list.index(max_prob)
        class_name = class_model.names[max_index]

        for painting in paintings:
            if painting["class"] == class_name:

                response: ChatResponse = chat(
                    model = llm_model, 
                    messages = [
                        {
                            'role': "user",
                            'content': f"You are a museum tour guide. I want you to describe the informations about a specific painting with the following data: Title: {painting["title"]}, Author: {painting["author"]}, Year: {painting["year"]}, Dimension: {painting["dimension"]}, Artistic Movement: {painting["artistic_movement"]}, Current Location: {painting["location"]}, Style: {painting["style"]}, Subject: {painting["subject"]}. Only English words are allowed"
                        }
                    ],
                    stream=True
                )

                if os.path.exists("./temp/tmp.txt"):
                    os.remove("./temp/tmp.txt")

                with open("./temp/tmp.txt", "a") as txtFile:
                    for chunk in response:
                        print(chunk['message']['content'], end='', flush=True)
                        txtFile.write(chunk["message"]["content"])

                loadTTS()

                print("\n")

                break
        
        
main()