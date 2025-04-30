

#%%
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio


def contrast_stretching(img, low_percent=5, high_percent=90):
    """
    Perform contrast stretching on a grayscale image based on given percentage range.
    
    Parameters:
        img (numpy.ndarray): Input grayscale image (2D numpy array).
        low_percent (float): The low percentage for contrast stretching (default is 5%).
        high_percent (float): The high percentage for contrast stretching (default is 90%).
    
    Returns:
        img_stretched (numpy.ndarray): Contrast stretched image.
        hist_stretched (numpy.ndarray): Histogram of the contrast stretched image.
    """
    # 計算直方圖與累積直方圖
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    cdf = np.cumsum(hist)
    cdf_normalized = cdf / cdf[-1]  # 正規化成 0~1

    # 根據百分比找到對應的灰階值
    low_cut = np.searchsorted(cdf_normalized, low_percent / 100.0)
    high_cut = np.searchsorted(cdf_normalized, high_percent / 100.0)

    # 進行線性拉伸並裁切
    img_stretched = np.clip((img - low_cut) * 255.0 / (high_cut - low_cut), 0, 255).astype(np.uint8)

    # 計算拉伸後的直方圖
    hist_stretched = cv2.calcHist([img_stretched], [0], None, [256], [0, 256])

    # ------- 圖片對比 -------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Grayscale Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_stretched, cmap='gray')
    plt.title(f"{low_percent}%-{high_percent}% Contrast Stretching")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # ------- 直方圖對比 -------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist, color='black')
    plt.title("Original Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])

    plt.subplot(1, 2, 2)
    plt.plot(hist_stretched, color='black')
    plt.title(f"Stretched Histogram ({low_percent}%~{high_percent}%)")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()

    # ------- 對映圖（mapping function） -------
    mapping = np.arange(256)
    mapped_value = np.clip((mapping - low_cut) * 255.0 / (high_cut - low_cut), 0, 255)

    plt.figure(figsize=(6, 5))
    plt.plot(mapping, mapped_value, color='blue')
    plt.title("Gray Level Mapping Function")
    plt.xlabel("Original Gray Level")
    plt.ylabel("Mapped Gray Level")
    plt.grid(True)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.axvline(low_cut, color='red', linestyle='--', label=f'low_cut = {low_cut}')
    plt.axvline(high_cut, color='green', linestyle='--', label=f'high_cut = {high_cut}')
    plt.legend()
    plt.show()

    return img_stretched, hist_stretched


def gamma_correction(img, gamma=1.4):
    """
    Apply Gamma correction to a grayscale image.
    
    Parameters:
        img (numpy.ndarray): Input grayscale image (2D numpy array).
        gamma (float): The gamma value for correction. 
                       A value > 1 will darken the image, while < 1 will brighten it. (default is 1.4).
    
    Returns:
        img_gamma (numpy.ndarray): Gamma corrected image.
        hist_gamma (numpy.ndarray): Histogram of the gamma corrected image.
    """
    # 正規化圖像到 0~1
    img_normalized = img / 255.0
    
    # 應用 Gamma 變換
    img_gamma = np.power(img_normalized, gamma)
    
    # 重新縮放回 0~255 並裁切
    img_gamma = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)
    
    # 計算拉伸後的直方圖
    hist_gamma = cv2.calcHist([img_gamma], [0], None, [256], [0, 256])

    # ---------- 顯示原圖與增強圖 ----------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Grayscale")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_gamma, cmap='gray')
    plt.title(f"Gamma Enhanced (γ={gamma})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # ---------- 直方圖比較 ----------
    hist_original = cv2.calcHist([img], [0], None, [256], [0, 256])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist_original, color='black')
    plt.title("Original Histogram")
    plt.xlim([0, 256])

    plt.subplot(1, 2, 2)
    plt.plot(hist_gamma, color='black')
    plt.title(f"Gamma Histogram (γ={gamma})")
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()

    # ---------- 對映圖 ----------
    x = np.linspace(0, 255, 256)
    y = 255 * (x / 255) ** gamma

    plt.figure(figsize=(6, 5))
    plt.plot(x, y, color='blue')
    plt.title(f"Gamma Mapping Function (γ={gamma})")
    plt.xlabel("Original Gray Level")
    plt.ylabel("Mapped Gray Level")
    plt.grid(True)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.show()

    return img_gamma, hist_gamma


def histogram_equalization(img):
    """
    Perform histogram equalization on a grayscale image and display the results.
    
    Parameters:
        img (numpy.ndarray): Input grayscale image (2D numpy array).
    
    Returns:
        equalized (numpy.ndarray): Histogram equalized image.
        hist_eq (numpy.ndarray): Histogram of the equalized image.
    """
    # 均衡化圖像
    equalized = cv2.equalizeHist(img)
    
    # 計算直方圖
    hist_orig = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256]).ravel()

    # ---------- 顯示原圖與均衡化後圖像 ----------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(equalized, cmap='gray')
    plt.title("Histogram Equalized")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # ---------- 直方圖比較 ----------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist_orig, color='black')
    plt.title("Original Histogram")

    plt.subplot(1, 2, 2)
    plt.plot(hist_eq, color='black')
    plt.title("Equalized Histogram")
    plt.tight_layout()
    plt.show()

    # ---------- 累積分佈函數 (CDF) 計算 ----------
    cdf = hist_orig.cumsum()
    cdf_normalized = hist_eq.cumsum()

    # 建立對映函數：將 CDF 轉成 [0,255] 並四捨五入為 int
    mapping = np.round(cdf_normalized * 255).astype(np.uint8)

    # 繪製對映圖
    plt.figure(figsize=(6, 5))
    plt.plot(mapping, color='blue')
    plt.title("Histogram Equalization Mapping")
    plt.xlabel("Original Gray Level")
    plt.ylabel("Mapped Gray Level")
    plt.grid(True)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.show()

    # ---------- 顯示原始 CDF 和正規化後的 CDF ----------
    plt.figure(figsize=(10, 6))

    # 顯示原始 CDF
    plt.plot(cdf, color='blue', label='Original CDF')

    # 顯示正規化後的 CDF
    plt.plot(cdf_normalized, color='green', label='Normalized CDF')

    plt.title("CDF Comparison: Original vs Normalized")
    plt.xlabel("Gray Level")
    plt.ylabel("Cumulative Frequency")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return equalized, hist_eq

#%%# 讀取灰階影像  Gray=0.299⋅R+0.587⋅G+0.114⋅B
img = cv2.imread("IMG_0893.jpg", cv2.IMREAD_GRAYSCALE)

a,b =contrast_stretching(img, low_percent=5, high_percent=90)
a,b =gamma_correction(img, gamma=1.4)
a,b =histogram_equalization(img)

img2 = cv2.imread("T51RUH_20250228T022539_TCI_10m.jpg",cv2.IMREAD_GRAYSCALE)
a,b =contrast_stretching(img2, low_percent=5, high_percent=90)
a,b =gamma_correction(img2, gamma=1.4)
a,b =histogram_equalization(img2)