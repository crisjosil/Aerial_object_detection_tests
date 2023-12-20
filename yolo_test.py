from ultralytics import YOLO
import matplotlib.pyplot as plt

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
print('Load a pretrained YOLO model (recommended for training)')
model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
#print('Train the model using the "coco128.yaml" dataset for 3 epochs')
#results = model.train(data='coco128.yaml', epochs=3)

# Evaluate the model's performance on the validation set
#print('Evaluate model')
#results = model.val()

# Perform object detection on an image using the model
#print('results')
#results = model('https://ultralytics.com/images/bus.jpg')
results = model('/home/crs2/Downloads/pexels-anjan-18407434.jpg',show=True, conf=0.4,save=True)
#print(results.shape)

fig,ax=plt.subplots()
ax.imshow(results)

# Export the model to ONNX format
#success = model.export(format='onnx')

