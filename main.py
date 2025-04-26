import cv2
import easyocr
import matplotlib.pyplot as plt

# Read image
img_path = r"E:\OCR\Data\test3.png"
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Error: Image not found at {img_path}")

# Instance text detector
reader = easyocr.Reader(['en'], gpu=False)

# Detect text on image
text_= reader.readtext(img)
threshold = 0.25

#Draw bbox
for t in text_:
    print(t)
    bbox,text,score = t

    if score > threshold:
        cv2.rectangle(img,bbox[0],bbox[2],(0,255,0),5)
        cv2.putText(img, text, bbox[0],cv2.FONT_HERSHEY_COMPLEX,0.65,(255,0,0),1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()