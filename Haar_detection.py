import sys
import cv2 as cv

i = 190
img = cv.imread("datasets/easy/images/original/N" + str(i) + ".jpeg")

grey = img[:,:,:]
cv.cvtColor(grey, cv.COLOR_BGR2GRAY)
haar = cv.CascadeClassifier("cascade/haarcascade_russian_plate_number.xml")

plates = haar.detectMultiScale(grey)

height, width = img.shape[:2]


def get_rect_coords(path):
    coords = []
    with open(path) as file:
        for line in file:
            coords.append(line.split(" ")[1:])
        return coords


coords = get_rect_coords("datasets/easy/yolo/N" + str(i) + ".txt")

for (x, y, w, h) in plates:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
for rect in range(len(coords)):
    x_center = float(coords[rect][0][:5])
    y_center = float(coords[rect][1][:5])
    x_w = float(coords[rect][2][:5])
    y_h = float(coords[rect][3][:5])

    cv.rectangle(img, (int((x_center - x_w / 2)*width), int((y_center - y_h / 2)*height)),
                      (int((x_center + x_w / 2)*width), int((y_center + y_h / 2)*height)), (0, 0, 255), 3)


cv.startWindowThread()
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()