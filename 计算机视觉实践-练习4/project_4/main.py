import cv2
import numpy as np

# 读取两张图片
img1 = cv2.imread('from.JPG', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('to.JPG', cv2.IMREAD_GRAYSCALE)

# 创建SIFT特征检测器
sift = cv2.SIFT.create()

# 检测特征点和计算描述子
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# 创建FLANN特征匹配器
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 匹配描述子
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 应用Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 至少需要4个匹配点来计算单应性矩阵
if len(good_matches) > 4:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 使用单应性矩阵对图像进行变换（例如图像拼接）
    height, width = img2.shape
    warped_img = cv2.warpPerspective(img1, H, (width, height))

    # 将两幅图像拼接在一起，创建一个足够大的画布
    canvas = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1]), dtype='uint8')
    canvas[:img1.shape[0], :img1.shape[1]] = img1
    canvas[:warped_img.shape[0], img1.shape[1]:] = warped_img

    # 保存结果
    cv2.imwrite('warped_image.jpg', warped_img)
    cv2.imwrite('canvas_with_warped_image.jpg', canvas)
else:
    print("Not enough matches found - %d/%d" % (len(good_matches), 4))
