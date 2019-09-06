import cv2


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    cap = cv2.VideoCapture(0)
    ret, image = cap.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # image = cv2.imread('picture/qianxun.jpg')
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    for (x, y, width, height) in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.imshow("Face", image)
    cv2.waitKey(1)


