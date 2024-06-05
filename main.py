import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = 'img6.png'  # Replace with the actual path to your image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detector
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw red border around detected particles
for contour in contours:
    cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)

# Save the result
output_path = 'particles_with_borders.png'  # Replace with the desired output path
cv2.imwrite(output_path, image)

# Display the result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

print(f"Output saved to {output_path}")
