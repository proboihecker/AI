from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("/home/garv/Projects/AI/competition/testimage.jpg")
results[0].show()