import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

# 1. 기본 텍스트 인식
def tesseract_basic_ocr(image_path):
    """기본 OCR 사용법"""
    # 이미지 로드
    image = cv2.imread(image_path)
    enhanced_image = enhance_image(image)
    
    
    # 텍스트 추출
    text = pytesseract.image_to_string(enhanced_image, lang='kor+eng')
    print("추출된 텍스트:")
    print(text)
    return text


def enhance_image(image):
    # 2x 크기 증가
    upscaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 샤프닝
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(upscaled, -1, kernel)
    
    return sharpened
    

#tesseract_basic_ocr("image/receipt.PNG")
tesseract_basic_ocr("image/writing_font_test.jpg")
#tesseract_basic_ocr("image/test2.jpg")
#tesseract_basic_ocr("image/recipt_example.jpg")
#tesseract_basic_ocr("image/korean_font_image.png")