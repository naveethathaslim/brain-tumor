import os
import cv2
import matplotlib.pyplot as plt

df = 'dataset5/'
cols = ['yes', 'no']
data1 = os.path.join(df, cols[0])

dt1 = os.listdir(data1)

def loading(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
    img = cv2.resize(img, (69, 69))
    return img


color_maps = ['gray', 'jet', 'viridis', 'plasma', 'inferno', 'magma']

plt.figure(figsize=(12, 9))

for i in range(min(6, len(dt1))):  
    img_path = os.path.join(data1, dt1[i])
    img = loading(img_path)
    
    for j, cmap in enumerate(color_maps):
        plt.subplot(6, len(color_maps), i * len(color_maps) + j + 1)
        plt.imshow(img, cmap=cmap)  
        plt.title(cmap)
        plt.axis('off')

plt.suptitle("Images with Different Color Maps", fontsize=16)
plt.tight_layout()
plt.show()
