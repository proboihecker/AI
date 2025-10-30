import cv2
import torch
import numpy as np
from torchvision import models, transforms
# from lama_cleaner.model_manager import get_model
# from lama_cleaner.helper import predict_mask

img = cv2.imread('testimage.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_tensor = transforms.ToTensor()(img)

model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

with torch.no_grad():
    predictions = model([img_tensor])[0]

threshold = 0.5
masks = predictions['masks'][predictions['scores'] > threshold]

combined_mask = torch.zeros_like(masks[0,0])
for mask in masks:
    combined_mask = torch.logical_or(combined_mask, mask[0] > 0.5)

mask_np = (combined_mask.cpu().numpy() * 255).astype("uint8")

# lama_model = get_model('lama')

full_mask = np.zeros(img.shape[:2], dtype="uint8")
for mask in masks:
    m = mask[0].cpu().numpy()
    full_mask[m > 0.5] = 255

result = cv2.inpaint(img, full_mask, 3, cv2.INPAINT_TELEA)

screen_res = 1920, 1080
scale_width = screen_res[0] / result.shape[1]
scale_height = screen_res[1] / result.shape[0]
scale = min(scale_width, scale_height)
window_width = int(result.shape[1] * scale)
window_height = int(result.shape[0] * scale)
res_img = cv2.resize(result, (window_width, window_height))

cv2.imshow("Inpainted Image", res_img)
cv2.waitKey(0)
cv2.destroyAllWindows()