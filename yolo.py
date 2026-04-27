from ultralytics import YOLO
from PIL import Image

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")



# Run inference with the YOLO11n model on the 'bus.jpg' image
results = model("image2.png")
print(results[0].to_json())


for i, r in enumerate(results):
    im_bgr = r.plot()
    im_rgb = Image.fromarray(im_bgr[..., ::-1])
    r.show()
    r.save(filename=f"results{i}.jpg")

