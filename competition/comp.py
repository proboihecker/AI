from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# FUNCTION TO GET HEX CODES
def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

# GETTING IMAGE ATTRIBUTES
img = Image.open("/home/garv/Projects/AI/competition/compimg.png").convert('RGB')
width, height = img.size
n_colors = 3
pixels = np.array(img).reshape(-1, 3)

# TO FIND DOMINANT COLORS
kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
kmeans.fit(pixels)
dominant_colors = kmeans.cluster_centers_
dominant_colors_uint8 = dominant_colors.astype(int)

# PLOTTING THE IMAGE
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img)
axes[0].set_title('Original Image')
axes[0].axis('off')

palette_image = np.zeros((50, 50 * n_colors, 3), dtype=np.uint8)
for i, color in enumerate(dominant_colors_uint8):
    palette_image[:, i * 50:(i + 1) * 50] = color

axes[1].imshow(palette_image)
axes[1].set_title('Dominant Colours')
axes[1].axis('off')

plt.show()

# SHOW HEX CODES OF EACH PIXEL
print(f"Hexadecimal color codes:")
for y in range(height):
    for x in range(width):
        r, g, b = img.getpixel((x, y))
        hex_code = rgb_to_hex(r, g, b)
        print(f"Pixel ({x}, {y}): {hex_code}")