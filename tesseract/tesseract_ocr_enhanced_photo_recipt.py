# 실제 영수증 이미지 전처리

import cv2
import numpy as np
from PIL import Image
import pytesseract

def preprocess_real_receipt(image_path, save_path=None):
    """실제 영수증 이미지 전처리 (스마트폰 촬영용)"""
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 불러올 수 없습니다!")
        return None
    
    h, w = image.shape[:2]
    print(f"원본 크기: {w} x {h}")
    
    # 1. 영수증 영역 검출 및 추출
    processed = extract_receipt_area(image)
    
    # 2. 조명 불균일 보정
    processed = correct_lighting(processed)
    
    # 3. 해상도 향상
    processed = enhance_resolution(processed)
    
    # 4. 텍스트 선명화
    processed = sharpen_text(processed)
    
    # 5. 최종 이진화
    final = final_binarization(processed)
    
    if save_path:
        cv2.imwrite(save_path, final)
        print(f"전처리된 이미지 저장: {save_path}")
    
    return final

def extract_receipt_area(image):
    """영수증 영역 자동 추출"""
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 엣지 검출
    edged = cv2.Canny(blurred, 50, 200, apertureSize=3)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    receipt_contour = None
    
    # 가장 큰 사각형 윤곽선 찾기
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # 4개 꼭지점이고 충분히 큰 영역
        if len(approx) == 4 and cv2.contourArea(contour) > (image.shape[0] * image.shape[1] * 0.1):
            receipt_contour = approx
            break
    
    if receipt_contour is not None:
        # 원근 변환
        warped = four_point_transform_safe(image, receipt_contour.reshape(4, 2))
        print("영수증 영역 자동 추출 성공")
        return warped
    else:
        print("영수증 윤곽선 검출 실패 - 원본 이미지 사용")
        return image

def four_point_transform_safe(image, pts):
    """안전한 4점 원근 변환"""
    
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 좌상단
        rect[2] = pts[np.argmax(s)]  # 우하단
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 우상단
        rect[3] = pts[np.argmax(diff)]  # 좌하단
        
        return rect
    
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # 너비와 높이 계산
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # 변환 후 좌표
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # 원근 변환
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def correct_lighting(image):
    """조명 불균일 보정"""
    
    # LAB 색공간으로 변환
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE로 조명 균일화
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # 다시 합치기
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def enhance_resolution(image):
    """해상도 향상"""
    
    h, w = image.shape[:2]
    
    # 너무 작으면 확대
    if h < 1500:
        scale = 1500 / h
        new_w = int(w * scale)
        new_h = 1500
        enhanced = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"해상도 향상: {scale:.2f}배 -> {new_w} x {new_h}")
    else:
        enhanced = image
        print("해상도 조정 없음")
    
    return enhanced

def sharpen_text(image):
    """텍스트 선명화"""
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 바이래터럴 필터 (엣지 보존하면서 노이즈 제거)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 언샤프 마스킹
    gaussian = cv2.GaussianBlur(filtered, (5, 5), 1.0)
    unsharp = cv2.addWeighted(filtered, 1.6, gaussian, -0.6, 0)
    
    # 대비 향상
    enhanced = cv2.convertScaleAbs(unsharp, alpha=1.2, beta=5)
    
    return enhanced

def final_binarization(image):
    """최종 이진화"""
    
    # 적응적 이진화
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # 형태학적 연산으로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def quick_receipt_preprocess(image_path, save_path=None):
    """빠른 영수증 전처리 (간단 버전)"""
    
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    print(f"빠른 전처리 시작: {w} x {h}")
    
    # 1. 크기 조정
    if h < 1200:
        scale = 1200 / h
        new_w = int(w * scale)
        resized = cv2.resize(image, (new_w, 1200), interpolation=cv2.INTER_CUBIC)
    else:
        resized = image
    
    # 2. 그레이스케일
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 3. 대비 향상
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 4. 노이즈 제거
    denoised = cv2.medianBlur(enhanced, 3)
    
    # 5. 샤프닝
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # 6. 이진화
    binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 13, 3)
    
    if save_path:
        cv2.imwrite(save_path, binary)
        print(f"빠른 전처리 완료: {save_path}")
    
    return binary

def receipt_ocr_with_preprocessing(image_path):
    """전처리 + OCR 통합 함수"""
    
    print("=== 영수증 OCR 처리 시작 ===")
    
    # 1. 전처리
    processed = preprocess_real_receipt(image_path, "processed_receipt.jpg")
    
    if processed is None:
        return None
    
    # 2. OCR 실행
    pil_image = Image.fromarray(processed)
    
    # 영수증 특화 OCR 설정
    config = '--psm 6 --oem 3'
    
    try:
        # 한국어 + 영어
        text = pytesseract.image_to_string(pil_image, lang='kor+eng', config=config)
        
        print("\n=== OCR 결과 ===")
        print(text)
        
        # 결과를 파일로도 저장
        with open("ocr_result.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("\nOCR 결과가 'ocr_result.txt'에 저장되었습니다.")
        
        return text
        
    except Exception as e:
        print(f"OCR 오류: {e}")
        return None

# 사용 예제
if __name__ == "__main__":
    
    # 이미지 경로 설정 (업로드한 이미지)
    image_path = "image/receipt.PNG"  # 실제 파일명으로 변경
    
    print("영수증 이미지 전처리 옵션:")
    print("1. 완전 자동 처리 (영역 검출 + 전처리 + OCR)")
    print("2. 고급 전처리만")
    print("3. 빠른 전처리만")
    
    # 1. 완전 자동 처리 (권장)
    #print("\n=== 완전 자동 처리 ===")
    #result_text = receipt_ocr_with_preprocessing(image_path)
    
    # 2. 전처리만 하고 싶다면
    #processed = preprocess_real_receipt(image_path, "image/output.jpg")
    
    # 3. 빠른 전처리만
    quick_processed = quick_receipt_preprocess(image_path, "image/quick_output.jpg")
    
    print("\n처리 완료!")