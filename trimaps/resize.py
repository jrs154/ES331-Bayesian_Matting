import cv2 as cv

image = []
dim = (300, 200)
x, y = 16, 18
for i in range(x, y):
    image.append(cv.imread("GT" + str(i) + ".png", 1))
print(len(image))
for i in range(x, y):
    filename = "S" + str(i) + ".png"
    img = cv.resize(image[i-x], dim, interpolation = cv.INTER_AREA)
    cv.imwrite(filename, img)
