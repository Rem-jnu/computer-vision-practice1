import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取左右图像
left_image = cv2.imread('left_image.JPG', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('right_image.JPG', cv2.IMREAD_GRAYSCALE)

# 对图像进行预处理
left_image = cv2.equalizeHist(left_image)
right_image = cv2.equalizeHist(right_image)


# 创建StereoSGBM对象
stereo = cv2.StereoSGBM.create(minDisparity=0,
                               numDisparities=16,
                               blockSize=5,
                               P1=8*3*5**2,
                               P2=32*3*5**2,
                               disp12MaxDiff=1,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=32)

# 计算视差图
disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0


# 归一化视差图到0-255范围以便显示
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# 显示视差图
plt.figure(figsize=(10, 5))
plt.imshow(disparity_normalized, cmap='gray')
plt.title('Disparity Map')
plt.axis('off')
# 保存视差图
cv2.imwrite('disparity_map.png', disparity_normalized)
plt.show()
