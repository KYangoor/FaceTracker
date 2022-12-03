import torch
import cv2

model = torch.hub.load('../modules/yolo', 'custom', path='../model/best.pt', source='local')

img = ''

results = model(img)

results.show()
