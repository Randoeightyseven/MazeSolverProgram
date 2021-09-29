#!/usr/bin/env python
import PySimpleGUI as sg
import os
from PIL import Image, ImageTk
import io
import cv2
import numpy as np
import utlis
import sys
import time


# Get the folder containing the images from the user
folder = sg.popup_get_folder('Image folder to open', default_path='')
if not folder:
    sg.popup_cancel('Cancelling')
    raise SystemExit()

# PIL supported image types
img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp")

# get list of files in folder
flist0 = os.listdir(folder)

# create sub list of image files (no sub folders, no wrong file types)
fnames = [f for f in flist0 if os.path.isfile(
    os.path.join(folder, f)) and f.lower().endswith(img_types)]

num_files = len(fnames)                # number of iamges found
if num_files == 0:
    sg.popup('No files in folder')
    raise SystemExit()

del flist0                             # no longer needed

# ------------------------------------------------------------------------------
# use PIL to read data of one image
# ------------------------------------------------------------------------------


def get_img_data(f, maxsize=(1200, 850), first=False):
    """Generate image data using PIL
    """
    img = Image.open(f)
    img.thumbnail(maxsize)
    if first:                     # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)
# ------------------------------------------------------------------------------


# make these 2 elements outside the layout as we want to "update" them later
# initialize to the first file in the list
filename = os.path.join(folder, fnames[0])  # name of first file in list
image_elem = sg.Image(data=get_img_data(filename, first=True))
filename_display_elem = sg.Text(filename, size=(80, 3))
file_num_display_elem = sg.Text('File 1 of {}'.format(num_files), size=(15, 1))

# define layout, show and read the form
col = [[filename_display_elem],
       [image_elem]]

col_files = [[sg.Listbox(values=fnames, change_submits=True, size=(60, 30), key='listbox')],
             [sg.Button('Next', size=(8, 2)), sg.Button('Prev', size=(8, 2)),
              sg.Button('Select', size=(8, 2)), sg.Button('Use Webcam', size=(12, 2)), file_num_display_elem]]

layout = [[sg.Column(col_files), sg.Column(col)]]

window = sg.Window('Image Browser', layout, return_keyboard_events=True,
                   location=(0, 0), use_default_focus=False)

# loop reading the user input and displaying image, filename
i = 0
while True:
    # read the form
    event, values = window.read()
    print(event, values)
    # perform button and keyboard operations
    if event == sg.WIN_CLOSED:
        break
    elif event in ('Next', 'MouseWheel:Down', 'Down:40', 'Next:34'):
        i += 1
        if i >= num_files:
            i -= num_files
        filename = os.path.join(folder, fnames[i])
    elif event in ('Prev', 'MouseWheel:Up', 'Up:38', 'Prior:33'):
        i -= 1
        if i < 0:
            i = num_files + i
        filename = os.path.join(folder, fnames[i])

    elif event in ('Select'):
        print(filename)
        break

    elif event in ("Use Webcam"):

        ########################################################################
        webCamFeed = True
        pathImage = "unformattedmaze.jpeg"
        cap = cv2.VideoCapture(0)
        cap.set(10, 160)
        heightImg = 640
        widthImg = 480
        ########################################################################

        utlis.initializeTrackbars()
        count = 0

        while True:

            if webCamFeed:
                success, img = cap.read()
            else:
                img = cv2.imread(pathImage)
            img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE
            imgBlank = np.zeros((heightImg, widthImg, 3),
                                np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
            imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
            thres = utlis.valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
            imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])  # APPLY CANNY BLUR
            kernel = np.ones((5, 5))
            imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # APPLY DILATION
            imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

            ## FIND ALL COUNTOURS
            imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
            imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
            contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
            cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS

            # FIND THE BIGGEST COUNTOUR
            biggest, maxArea = utlis.biggestContour(contours)  # FIND THE BIGGEST CONTOUR
            if biggest.size != 0:
                biggest = utlis.reorder(biggest)
                cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
                imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
                pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
                pts2 = np.float32(
                    [[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

                # REMOVE 20 PIXELS FORM EACH SIDE
                imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
                imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

                # APPLY ADAPTIVE THRESHOLD
                imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
                imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
                imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
                imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

                # Image Array for Display
                imageArray = ([img, imgGray, imgThreshold, imgContours],
                              [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])

            else:
                imageArray = ([img, imgGray, imgThreshold, imgContours],
                              [imgBlank, imgBlank, imgBlank, imgBlank])

            # LABELS FOR DISPLAY
            lables = [["Original", "Gray", "Threshold", "Contours"],
                      ["Biggest Contour", "Warp Prespective", "Warp Gray", "Adaptive Threshold"]]

            stackedImage = utlis.stackImages(imageArray, 0.75, lables)
            cv2.imshow("Result", stackedImage)

            # SAVE IMAGE WHEN 's' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.imwrite("Image folder/custommaze.jpeg", imgAdaptiveThre)
                cv2.rectangle(stackedImage,
                              ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                              (1100, 350), (0, 255, 0), cv2.FILLED)
                cv2.putText(stackedImage, "Scan Saved",
                            (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                            cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
                cv2.imshow('Result', stackedImage)
                cv2.waitKey(300)
                count += 1
                print("Restart the main file")
                time.sleep(3)
                sys.exit()


    elif event == 'listbox':            # something from the listbox
        f = values["listbox"][0]            # selected filename
        filename = os.path.join(folder, f)  # read this file
        i = fnames.index(f)                 # update running index
    else:
        filename = os.path.join(folder, fnames[i])

    # update window with new image
    image_elem.update(data=get_img_data(filename, first=True))
    # update window with filename
    filename_display_elem.update(filename)
    # update page display
    file_num_display_elem.update('File {} of {}'.format(i+1, num_files))

window.close()
