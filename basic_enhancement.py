import cv2
import numpy as np

input_path = "images/aerial1.jpg"

img = cv2.imread(input_path)
img_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)


# resolution enhancement
scale_x = 2
scale_y = 2
# bicubic
bicubic = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
cv2.imwrite("images/bicubic.jpg", bicubic)

# contrast enhancement
# general histogram equalisation

ghe = cv2.equalizeHist(img_gray)
cv2.imwrite("images/ghe.jpg", ghe)

# adaptive histogram equalisation
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
ahe = clahe.apply(img_gray)
cv2.imwrite("images/ahe.jpg", ahe)

# min-max contrast stretching
minmax = ((img_gray - np.amin(img_gray)) / (np.amax(img_gray) - np.amin(img_gray))) * 255
cv2.imwrite("images/minmax.jpg", minmax)

# edge enhancement

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
auto_edge = auto_canny(blurred)
cv2.imwrite("images/edge.jpg", auto_edge)
