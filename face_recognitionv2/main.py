import cv2
import face_recognition

per1=face_recognition.load_image_file("obama.jpg")
per1encodings=face_recognition.face_encodings(per1)[0]

per2=face_recognition.load_image_file("biden.jpg")
per2encodings=face_recognition.face_encodings(per2)[0]

encodinglist=[per1encodings,per2encodings]
namelist=["obama","biden"]

image=cv2.imread("bidenliobama.jpg")
test1=face_recognition.load_image_file("bidenliobama.jpg")
facelocations=face_recognition.face_locations(test1)
faceencodings=face_recognition.face_encodings(test1,facelocations)

for faceloc,faceencoding in zip(facelocations,faceencodings):
    ustsoly,altsagx,altsagy,ustsolx=faceloc
    matchfaces=face_recognition.compare_faces(encodinglist,faceencoding)

    name="unknown"

    if True in matchfaces:
        matchedindex=matchfaces.index(True)
        name=namelist[matchedindex]

    cv2.rectangle(image,(ustsolx,ustsoly),(altsagx,altsagy),(255,0,0),2)

    cv2.putText(image,name,(ustsolx,ustsoly),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Window",image)
    print(name)

cv2.waitKey(0)
cv2.destroyAllWindows()


