import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract


# 1. 기본 전처리 함수들
def resize_image(image, scale_factor=2.0):
    """이미지 크기 조정 (OCR 정확도 향상을 위해 확대)"""
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

def convert_to_grayscale(image):
    """그레이스케일 변환"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def remove_noise(image, method='gaussian'):
    """노이즈 제거"""
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'median':
        return cv2.medianBlur(image, 5)
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    else:
        return image

def threshold_image(image, method='otsu'):
    """이진화 (흑백 변환)"""
    if method == 'otsu':
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive_gaussian':
        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    elif method == 'adaptive_mean':
        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    else:  # 고정 임계값
        _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    return thresh

# 2. 고급 전처리 함수들
def sharpen_image(image, strength=1.0):
    """이미지 샤프닝"""
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1], 
                       [-1,-1,-1]]) * strength
    return cv2.filter2D(image, -1, kernel)

def enhance_contrast(image, alpha=1.5, beta=0):
    """대비 향상"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def gamma_correction(image, gamma=1.2):
    """감마 보정"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                     for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def histogram_equalization(image):
    """히스토그램 평활화"""
    if len(image.shape) == 3:
        # 컬러 이미지의 경우
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    else:
        # 그레이스케일 이미지의 경우
        return cv2.equalizeHist(image)

def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8,8)):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(image)

# 3. 텍스트 특화 전처리
def detect_and_correct_skew(image):
    """기울어진 텍스트 보정"""
    # 텍스트 라인 검출
    gray = convert_to_grayscale(image)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is not None:
        # 각도 계산
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            angles.append(angle)
        
        # 중간값 각도로 회전
        median_angle = np.median(angles)
        if median_angle < 45:
            rotation_angle = median_angle
        else:
            rotation_angle = median_angle - 90
        
        # 이미지 회전
        if abs(rotation_angle) > 0.5:  # 0.5도 이상일 때만 보정
            center = (image.shape[1]//2, image.shape[0]//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            return cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), 
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return image

def morphological_operations(image, operation='opening', kernel_size=2):
    """형태학적 연산 (노이즈 제거, 텍스트 연결)"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    if operation == 'opening':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    elif operation == 'erosion':
        return cv2.erode(image, kernel, iterations=1)
    elif operation == 'dilation':
        return cv2.dilate(image, kernel, iterations=1)
    else:
        return image

def remove_borders(image, border_size=10):
    """이미지 테두리 제거"""
    h, w = image.shape[:2]
    return image[border_size:h-border_size, border_size:w-border_size]

# 4. 통합 전처리 함수들
def basic_preprocessing(image_path, save_path=None):
    """기본 전처리 파이프라인"""
    # 이미지 로드
    image = cv2.imread(image_path)
    
    # 1. 크기 조정 (2배 확대)
    resized = resize_image(image, scale_factor=2.0)
    
    # 2. 그레이스케일 변환
    gray = convert_to_grayscale(resized)
    
    # 3. 노이즈 제거
    denoised = remove_noise(gray, method='gaussian')
    
    # 4. 이진화
    binary = threshold_image(denoised, method='otsu')
    
    if save_path:
        cv2.imwrite(save_path, binary)
    
    return binary

def advanced_preprocessing(image_path, save_path=None):
    """고급 전처리 파이프라인"""
    # 이미지 로드
    image = cv2.imread(image_path)
    
    # 1. 테두리 제거
    no_border = remove_borders(image, border_size=5)
    
    # 2. 크기 조정
    resized = resize_image(no_border, scale_factor=2.5)
    
    # 3. 그레이스케일 변환
    gray = convert_to_grayscale(resized)
    
    # 4. CLAHE로 대비 향상
    enhanced = clahe_enhancement(gray)
    
    # 5. 노이즈 제거
    denoised = remove_noise(enhanced, method='bilateral')
    
    # 6. 샤프닝
    sharpened = sharpen_image(denoised, strength=0.5)
    
    # 7. 적응적 이진화
    binary = threshold_image(sharpened, method='adaptive_gaussian')
    
    # 8. 형태학적 연산 (노이즈 제거)
    cleaned = morphological_operations(binary, operation='opening', kernel_size=1)
    
    if save_path:
        cv2.imwrite(save_path, cleaned)
    
    return cleaned

def document_preprocessing(image_path, save_path=None):
    """문서 이미지 특화 전처리"""
    image = cv2.imread(image_path)
    
    # 1. 기울기 보정
    corrected = detect_and_correct_skew(image)
    
    # 2. 크기 조정
    resized = resize_image(corrected, scale_factor=2.0)
    
    # 3. 그레이스케일
    gray = convert_to_grayscale(resized)
    
    # 4. 히스토그램 평활화
    equalized = histogram_equalization(gray)
    
    # 5. 가우시안 블러
    blurred = remove_noise(equalized, method='gaussian')
    
    # 6. 이진화
    binary = threshold_image(blurred, method='otsu')
    
    # 7. 형태학적 연산
    cleaned = morphological_operations(binary, operation='closing', kernel_size=1)
    
    if save_path:
        cv2.imwrite(save_path, cleaned)
    
    return cleaned

def handwriting_preprocessing(image_path, save_path=None):
    """손글씨 이미지 전처리"""
    image = cv2.imread(image_path)
    
    # 1. 크기 조정 (손글씨는 더 크게)
    resized = resize_image(image, scale_factor=3.0)
    
    # 2. 그레이스케일
    gray = convert_to_grayscale(resized)
    
    # 3. 감마 보정
    gamma_corrected = gamma_correction(gray, gamma=1.2)
    
    # 4. CLAHE
    enhanced = clahe_enhancement(gamma_corrected, clip_limit=3.0)
    
    # 5. 바이래터럴 필터 (엣지 보존하면서 노이즈 제거)
    filtered = remove_noise(enhanced, method='bilateral')
    
    # 6. 적응적 이진화
    binary = threshold_image(filtered, method='adaptive_gaussian')
    
    # 7. 형태학적 연산 (연결된 글자 분리)
    cleaned = morphological_operations(binary, operation='opening', kernel_size=1)
    
    if save_path:
        cv2.imwrite(save_path, cleaned)
    
    return cleaned

# 5. 전처리 결과 비교 함수
def compare_preprocessing_methods(image_path, save_comparison=True):
    """다양한 전처리 방법 비교"""
    original = cv2.imread(image_path)
    
    methods = {
        'Original': original,
        'Basic': basic_preprocessing(image_path),
        'Advanced': advanced_preprocessing(image_path),
        'Document': document_preprocessing(image_path),
        'Handwriting': handwriting_preprocessing(image_path)
    }
    
    # 각 방법별로 이미지 저장
    if save_comparison:
        for name, img in methods.items():
            filename = f"comparison_{name.lower()}.jpg"
            cv2.imwrite(filename, img)
            print(f"저장됨: {filename}")
   
    
    return methods

# 6. 실용적인 올인원 함수
def smart_preprocessing(image_path, text_type='mixed', save_path=None):
    """
    텍스트 타입에 따른 스마트 전처리
    
    Args:
        image_path: 입력 이미지 경로
        text_type: 'document', 'handwriting', 'mixed', 'photo'
        save_path: 저장 경로 (선택사항)
    """
    
    if text_type == 'document':
        result = document_preprocessing(image_path, save_path)
    elif text_type == 'handwriting':
        result = handwriting_preprocessing(image_path, save_path)
    elif text_type == 'photo':
        result = advanced_preprocessing(image_path, save_path)
    else:  # mixed
        result = basic_preprocessing(image_path, save_path)
    
    print(f"전처리 완료: {text_type} 타입으로 처리됨")
    return result


# 1. 기본 텍스트 인식
def basic_ocr(image_path):
    """기본 OCR 사용법"""
    # 이미지 로드
    image = Image.open(image_path)
    #image = handwriting_preprocessing(image_path, "image/enhanced_image.jpg")
    
    
    # 텍스트 추출
    text = pytesseract.image_to_string(image, lang='kor+eng')
    print("추출된 텍스트:")
    print(text)
    return text

# 2. 한국어 특화 OCR
def korean_ocr(image_path):
    """한국어 OCR 최적화"""
    image = Image.open(image_path)
    
    # 한국어 + 영어 설정
    custom_config = r'--oem 3 --psm 6 -l kor+eng'
    text = pytesseract.image_to_string(image, config=custom_config)
    
    print("한국어 OCR 결과:")
    print(text)
    return text

# 3. 이미지 전처리 + OCR
def preprocessed_ocr(image_path):
    """이미지 전처리 후 OCR"""
    # OpenCV로 이미지 로드
    img = cv2.imread(image_path)
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거
    denoised = cv2.medianBlur(gray, 5)
    
    # 이진화 (임계값 처리)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # PIL Image로 변환
    pil_image = Image.fromarray(thresh)
    
    # OCR 실행
    text = pytesseract.image_to_string(pil_image, lang='kor+eng')
    
    print("전처리 후 OCR 결과:")
    print(text)
    return text

# 4. 바운딩 박스와 함께 텍스트 추출
def ocr_with_boxes(image_path):
    """텍스트 위치 정보와 함께 추출"""
    image = Image.open(image_path)
    
    # 텍스트와 바운딩 박스 정보 추출
    data = pytesseract.image_to_data(image, lang='kor+eng', output_type=pytesseract.Output.DICT)
    
    print("텍스트와 위치 정보:")
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60:  # 신뢰도 60% 이상만
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            text = data['text'][i].strip()
            if text:
                print(f"텍스트: '{text}', 위치: ({x}, {y}, {w}, {h}), 신뢰도: {data['conf'][i]}")
    
    return data

# 5. 다양한 PSM 모드 테스트
def test_psm_modes(image_path):
    """다양한 Page Segmentation Mode 테스트"""
    image = Image.open(image_path)
    
    psm_modes = {
        3: "Fully automatic page segmentation, but no OSD",
        6: "Uniform block of text",
        7: "Single text line",
        8: "Single word",
        11: "Sparse text",
        13: "Raw line. Treat the image as a single text line"
    }
    
    print("다양한 PSM 모드 테스트:")
    for psm, description in psm_modes.items():
        config = f'--oem 3 --psm {psm} -l kor+eng'
        try:
            text = pytesseract.image_to_string(image, config=config)
            print(f"\nPSM {psm} ({description}):")
            print(text[:100] + "..." if len(text) > 100 else text)
        except Exception as e:
            print(f"PSM {psm} 오류: {e}")

# 6. 이미지 품질 향상 함수들
def enhance_image_quality(image_path):
    """이미지 품질 향상"""
    img = cv2.imread(image_path)
    
    # 1. 크기 조정 (OCR 정확도 향상을 위해 2배 확대)
    height, width = img.shape[:2]
    img_resized = cv2.resize(img, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    
    # 2. 그레이스케일 변환
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 3. 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. 샤프닝
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    
    # 5. 적응적 임계값 처리
    adaptive_thresh = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # PIL Image로 변환
    pil_image = Image.fromarray(adaptive_thresh)
    
    # OCR 실행
    text = pytesseract.image_to_string(pil_image, lang='kor+eng', config='--oem 3 --psm 6')
    
    print("이미지 품질 향상 후 OCR:")
    print(text)
    return text

# 7. 실용적인 OCR 함수 (권장)
def smart_ocr(image_path, lang='kor+eng'):
    """실용적인 OCR 함수"""
    try:
        # 여러 방법으로 시도
        image = Image.open(image_path)
        
        # 방법 1: 기본 설정
        config1 = '--oem 3 --psm 6'
        text1 = pytesseract.image_to_string(image, lang=lang, config=config1)
        
        # 방법 2: 전처리 후
        img_cv = cv2.imread(image_path)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pil_thresh = Image.fromarray(thresh)
        text2 = pytesseract.image_to_string(pil_thresh, lang=lang, config=config1)
        
        # 더 긴 결과 선택
        result = text1 if len(text1) > len(text2) else text2
        
        print("Smart OCR 결과:")
        print(result)
        return result
        
    except Exception as e:
        print(f"OCR 오류: {e}")
        return ""

# 8. 사용 예제
if __name__ == "__main__":
    # 이미지 파일 경로 (실제 경로로 변경하세요)
    image_path = "image/computer_font_korean.png"
    
    print("=== Tesseract OCR 테스트 ===\n")
    
    # 기본 OCR
    #basic_ocr(image_path)
    #print("\n" + "="*50 + "\n")
    
    # 한국어 특화 OCR
    #korean_ocr(image_path)
    #print("\n" + "="*50 + "\n")
    
    # 전처리 + OCR
    #preprocessed_ocr(image_path)
    #print("\n" + "="*50 + "\n")
    
    # 바운딩 박스 정보
    #ocr_with_boxes(image_path)
    #print("\n" + "="*50 + "\n")
    
    # PSM 모드 테스트
    #test_psm_modes(image_path)
    #print("\n" + "="*50 + "\n")
    
    # 실용적인 OCR
    smart_ocr(image_path)