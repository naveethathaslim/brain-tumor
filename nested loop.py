import os
import cv2
import matplotlib.pyplot as plt


dataset_path = 'dataset5/'
categories = ['yes', 'no']


def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (69, 69))
    return img[..., ::-1]  


plt.figure(figsize=(12, 9))

for idx, category in enumerate(categories):
    folder_path = os.path.join(dataset_path, category)
    images = os.listdir(folder_path)

    
    for i in range(min(6, len(images))):
        img_path = os.path.join(folder_path, images[i])
        plt.subplot(2, 6, idx * 6 + i + 1)  
        plt.imshow(load_image(img_path))
        plt.title(category)
        plt.axis('off')

plt.suptitle("Images from 'yes' and 'no' folders", fontsize=16)
plt.tight_layout()
plt.show()
