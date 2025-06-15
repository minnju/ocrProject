import easyocr
import cv2
import numpy as np

# EasyOCR 초기화

def easyOcr(image_path):

    reader = easyocr.Reader(['ko', 'en'], gpu=True)  # GPU 사용시 더 빠름
    image = cv2.imread(image_path)
    enhanced_image=enhance_image(image)
    # 손글씨 이미지 읽기
    result = reader.readtext(enhanced_image,
                            detail=1,
                            paragraph=False,
                            min_size=10,      # 최소 텍스트 크기
                            text_threshold=0.7,  # 텍스트 검출 임계값
                            low_text=0.4,     # 낮은 신뢰도 텍스트도 포함
                            link_threshold=0.4,  # 문자 연결 임계값
                            canvas_size=2560,  # 캔버스 크기 증가
                            mag_ratio=1.5)     # 확대 비율

    # 결과 출력
    for (bbox, text, confidence) in result:
        print(f"{text}")
        #print("신뢰도: {confidence:.2f}")

    # 해상도 증가
def enhance_image(image):
    # 2x 크기 증가
    upscaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 샤프닝
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(upscaled, -1, kernel)
    
    return sharpened

#easyOcr("test2.jpg")
#easyOcr("writing_font_test.jpg")
#easyOcr("computer_font_korean.png")
#easyOcr("receipt.PNG")
easyOcr("image/recipt_example.jpg")