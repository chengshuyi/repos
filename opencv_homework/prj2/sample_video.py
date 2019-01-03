import cv2
capture = cv2.VideoCapture(0)
capture.set(3,640)
capture.set(4,480)
capture.set(1, 10.0)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('./output.avi', fourcc,30,(640,480))
while True:
    ret,frame = capture.read()
    if ret is True:
        out.write(frame)
        cv2.imshow('windows',frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break
capture.release()
out.release()
cv2.destroyAllWindows()
