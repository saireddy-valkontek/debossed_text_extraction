import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image


def recText(filename):
    img = cv2.imread("light.jpg")
    img = cv2.resize(img, (350, 80))
    cv2.imshow('Original', img)

    dst = cv2.fastNlMeansDenoisingColored(img, None, 12, 12, 7, 21)
    cv2.imshow('denoise', dst)

    '''
    kernel=np.ones((5,5),np.uint8)
    erosion=cv2.erode(img,kernel,iterations=1)
    cv2.imshow('Erosion.jpg', erosion)

    height, width = sharp.shape[:2]
    # get the center of the image
    center_x, center_y = (width/2, height/2)
    # rotate the image by 60 degrees counter-clockwise around the center of the image
    M = cv2.getRotationMatrix2D((center_x, center_y), 270, 1.0)
    rotated_image = cv2.warpAffine(sharp, M, (width, height))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 250, 0), 2)

    cv2.imshow('rota.jpg', img)
    '''  # Deboss filter kernel variation 2
    kernel_deboss_2 = np.array([[0, -2, -2],
                                [2, 0, -2],
                                [2, 2, 0]])

    # Deboss filter kernel variation 3
    kernel_deboss_3 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    kernel_deboss_5 = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])

    kernel_deboss_6 = np.array([[-2, -1, 0],
                                [-1, 1, 1],
                                [0, 1, 2]])

    debossed_image_8U5 = cv2.filter2D(dst, cv2.CV_8U, kernel_deboss_5, borderType=cv2.BORDER_REFLECT)
    cv2.imshow('8Ude5.jpg', debossed_image_8U5)

    debossed_image_16S5 = cv2.filter2D(dst, -1, kernel_deboss_5, borderType=cv2.BORDER_REFLECT)
    cv2.imshow('16Sde5.jpg', debossed_image_16S5)

    debossed_image_32F5 = cv2.filter2D(dst, cv2.CV_32F, kernel_deboss_5, borderType=cv2.BORDER_REFLECT)
    cv2.imshow('32Fde5.jpg', debossed_image_32F5)

    debossed_image_64F5 = cv2.filter2D(dst, cv2.CV_64F, kernel_deboss_5, borderType=cv2.BORDER_REFLECT)
    cv2.imshow('64Fde5.jpg', debossed_image_64F5)
    # -----------------------------------------------------------------------------------------------------------------
    debossed_image_8U6 = cv2.filter2D(dst, cv2.CV_8U, kernel_deboss_6, borderType=cv2.BORDER_REFLECT)
    cv2.imshow('8Ude6.jpg', debossed_image_8U6)

    debossed_image_16S6 = cv2.filter2D(dst, -1, kernel_deboss_6, borderType=cv2.BORDER_REFLECT)
    cv2.imshow('16Sde6.jpg', debossed_image_16S6)

    debossed_image_32F6 = cv2.filter2D(dst, cv2.CV_32F, kernel_deboss_6, borderType=cv2.BORDER_REFLECT)
    cv2.imshow('32Fde6.jpg', debossed_image_32F6)

    debossed_image_64F6 = cv2.filter2D(dst, cv2.CV_64F, kernel_deboss_6, borderType=cv2.BORDER_REFLECT)
    cv2.imshow('64Fde6.jpg', debossed_image_64F6)
    # ------------------------------------------------------------------------------------------------------------------
    debossed_image_8U3 = cv2.filter2D(dst, cv2.CV_8U, kernel_deboss_3, borderType=cv2.BORDER_REFLECT)
    cv2.imshow('8Ude3.jpg', debossed_image_8U3)

    debossed_image_16S3 = cv2.filter2D(dst, -1, kernel_deboss_3, borderType=cv2.BORDER_REFLECT)
    cv2.imshow('16Sde3.jpg', debossed_image_16S3)

    debossed_image_32F3 = cv2.filter2D(dst, cv2.CV_32F, kernel_deboss_3, borderType=cv2.BORDER_REFLECT)
    cv2.imshow('32Fde3.jpg', debossed_image_32F3)

    debossed_image_64F3 = cv2.filter2D(dst, cv2.CV_64F, kernel_deboss_3, borderType=cv2.BORDER_REFLECT)
    cv2.imshow('64Fde3.jpg', debossed_image_64F3)
    # ------------------------------------------------------------------------------------------------------------------
    # Apply debossing filter with CV_8U depth
    debossed_image_8U = cv2.filter2D(dst, cv2.CV_8U, kernel_deboss_2, borderType=cv2.BORDER_REFLECT)
    cv2.imshow('8U.jpg', debossed_image_8U)

    # Apply debossing filter with CV_16S depth
    debossed_image_16S = cv2.filter2D(dst, -1, kernel_deboss_2, borderType=cv2.BORDER_REFLECT)
    cv2.imshow('16S.jpg', debossed_image_16S)

    # Apply debossing filter with CV_32F depth
    debossed_image_32F = cv2.filter2D(dst, cv2.CV_32F, kernel_deboss_2, borderType=cv2.BORDER_REFLECT)
    cv2.imshow('32F.jpg', debossed_image_32F)

    # Apply debossing filter with CV_64F depth
    debossed_image_64F = cv2.filter2D(dst, cv2.CV_64F, kernel_deboss_2, borderType=cv2.BORDER_REFLECT)
    cv2.imshow('64F.jpg', debossed_image_64F)

    '''kernel = np.ones((5,5),np.float32)/25
    img_blur = cv2.filter2D(img,-1,kernel)
    cv2.imshow("blur", img_blur)

    img_blur = cv2.bilateralFilter(img,9,75,75)
    img_blur = cv2.GaussianBlur(img, (3,3), 0)
    cv2.imshow('Blur', img_blur)

    kernel_sharpening = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharp = cv2.filter2D(dst, -9, kernel_sharpening)
    cv2.imshow('sharp.jpg', sharp) 

    th2 = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imshow('thresh2', th2)
    th3 = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)    
    cv2.imshow('thresh3', th3)
    _, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh1', th1)
    '''
    # Canny Edge Detection
    edges = cv2.Canny(dst, 100, 200)  # Canny Edge Detection
    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges)

    bedges = cv2.Canny(dst, threshold1=160, threshold2=280)  # Canny Edge Detection
    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection2', bedges)

    '''    
    # Otsu's thresholding
    _,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("OTSUNOGAUS", th2)

    _,th3 = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("OTSU", th3)
    ret, thresh1 = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)
    contours2, hierarchy2 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image_copy2 = img.copy()
    cv2.drawContours(image_copy2, contours2, -1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('SIMPLE Approximation contours', image_copy2)

    image_copy3 = img.copy()
    for i, contour in enumerate(contours2): # loop over one contour area
       for j, contour_point in enumerate(contour): # loop over the points
           # draw a circle on the current contour coordinate
           cv2.circle(image_copy3, ((contour_point[0][0], contour_point[0][1])), 2, (0, 255, 0), 2, cv2.LINE_AA)
    # see the results
    cv2.imshow('CHAIN_APPROX_SIMPLE Point only', image_copy3)
    '''
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    txt = pytesseract.image_to_string(Image.open(filename), lang='eng', config='--psm 13')
    return txt


info = recText('light.jpg')
print(info)
file = open("new.txt", "w")
file.write(info)
file.close()
print("Written Successfully")