# 영수증 특화 이미지 전처리 함수

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import math
import pytesseract

def detect_receipt_contour(image):
    """영수증 윤곽선 검출 및 추출"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러 적용
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 엣지 검출
    edged = cv2.Canny(blurred, 75, 200)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 가장 큰 사각형 윤곽선 찾기
    receipt_contour = None
    for contour in contours:
        # 윤곽선 근사화
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # 4개의 꼭지점을 가진 윤곽선이고 충분히 큰 경우
        if len(approx) == 4 and cv2.contourArea(contour) > 1000:
            receipt_contour = approx
            break
    
    return receipt_contour

def four_point_transform(image, pts):
    """4점 원근 변환으로 영수증 정렬 (크기 보존)"""
    # 점들을 정렬 (top-left, top-right, bottom-right, bottom-left)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # 새로운 이미지의 너비와 높이 계산
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # 최소 크기 보장 (너무 작아지지 않도록)
    original_h, original_w = image.shape[:2]
    maxWidth = max(maxWidth, int(original_w * 0.5))
    maxHeight = max(maxHeight, int(original_h * 0.5))
    
    # 변환 후 좌표
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # 원근 변환 매트릭스 계산 및 적용
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def order_points(pts):
    """점들을 시계방향으로 정렬"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # 좌상단: 합이 가장 작은 점, 우하단: 합이 가장 큰 점
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # 우상단: 차이가 가장 작은 점, 좌하단: 차이가 가장 큰 점
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def correct_skew_angle(image):
    """영수증 기울기 자동 보정 (크기 보존)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 엣지 검출
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 허프 변환으로 직선 검출
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is not None:
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            # 수직/수평에 가까운 선들만 고려
            if angle < 45 or angle > 135:
                if angle > 135:
                    angle = angle - 180
                angles.append(angle)
        
        if angles:
            # 중간값으로 회전 각도 결정
            rotation_angle = np.median(angles)
            
            # 각도가 너무 작으면 보정하지 않음
            if abs(rotation_angle) > 0.5:
                # 회전으로 인한 크기 증가 계산
                h, w = image.shape[:2]
                rad = math.radians(abs(rotation_angle))
                new_w = int(w * math.cos(rad) + h * math.sin(rad))
                new_h = int(h * math.cos(rad) + w * math.sin(rad))
                
                # 회전 중심점
                center = (w//2, h//2)
                M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                
                # 변환 행렬 조정 (이미지가 잘리지 않도록)
                M[0, 2] += (new_w - w) // 2
                M[1, 2] += (new_h - h) // 2
                
                return cv2.warpAffine(image, M, (new_w, new_h), 
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return image

def enhance_receipt_contrast(image):
    """영수증 대비 향상 (열전사 프린터 대응)"""
    # LAB 색공간으로 변환
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE 적용 (적응적 히스토그램 평활화)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # 다시 합치기
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def remove_receipt_noise(image):
    """영수증 특유의 노이즈 제거"""
    # 1. 미디언 필터로 점 노이즈 제거
    denoised = cv2.medianBlur(image, 3)
    
    # 2. 바이래터럴 필터로 엣지 보존하면서 노이즈 제거
    denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
    
    # 3. 모폴로지 연산으로 텍스트 보강
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    
    return denoised

def thermal_printer_enhancement(image):
    """열전사 프린터 영수증 특화 처리"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. 감마 보정 (어두운 부분 밝게)
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                     for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(gray, table)
    
    # 2. 언샤프 마스킹 (텍스트 선명도 향상)
    gaussian = cv2.GaussianBlur(gamma_corrected, (0, 0), 2.0)
    unsharp_mask = cv2.addWeighted(gamma_corrected, 2.0, gaussian, -1.0, 0)
    
    # 3. 적응적 이진화
    binary = cv2.adaptiveThreshold(unsharp_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    return binary

def receipt_line_detection_and_removal(image):
    """영수증의 구분선 검출 및 제거 (선택적)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # 수평선 검출을 위한 커널
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # 검출된 선 제거
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    result = gray.copy()
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255, 255, 255), 2)
    
    return result

def crop_receipt_margins(image, margin_ratio=0.02):
    """영수증 여백 자동 제거 (안전한 크롭)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # 이진화
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 텍스트 영역 찾기
    coords = np.column_stack(np.where(binary > 0))
    
    if len(coords) > 0:
        # 바운딩 박스 계산
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # 여백 추가
        h, w = image.shape[:2]
        margin_h = int(h * margin_ratio)
        margin_w = int(w * margin_ratio)
        
        y_min = max(0, y_min - margin_h)
        y_max = min(h, y_max + margin_h)
        x_min = max(0, x_min - margin_w)
        x_max = min(w, x_max + margin_w)
        
        # 크기 검증 (너무 작아지지 않도록)
        crop_h = y_max - y_min
        crop_w = x_max - x_min
        
        if crop_h > h * 0.3 and crop_w > w * 0.3:  # 원본의 30% 이상 유지
            return image[y_min:y_max, x_min:x_max]
        else:
            print("크롭 영역이 너무 작아 원본 이미지 유지")
    
    return image

def receipt_preprocessing_sharp(image_path, save_path=None):
    """선명도 우선 영수증 전처리"""
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    print(f"원본 크기: {w} x {h}")
    
    # 1. 해상도 향상 (무조건 확대)
    if h < 1500:
        scale = 1500 / h
        new_w = int(w * scale)
        new_h = 1500
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"확대: {scale:.2f}배 -> {new_w} x {new_h}")
    else:
        resized = image
        new_w, new_h = w, h
    
    # 2. 그레이스케일 변환
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 3. 대비 강화 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 4. 가우시안 블러 (약간)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # 5. 언샤프 마스킹 (선명화)
    unsharp_mask = cv2.addWeighted(enhanced, 1.8, blurred, -0.8, 0)
    
    # 6. 감마 보정
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(unsharp_mask, table)
    
    # 7. 적응적 이진화 (파라미터 조정)
    binary = cv2.adaptiveThreshold(gamma_corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 15, 4)
    
    # 8. 형태학적 연산 (미세 조정)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    print(f"최종 크기: {cleaned.shape[1]} x {cleaned.shape[0]}")
    
    if save_path:
        cv2.imwrite(save_path, cleaned)
        print(f"선명한 이미지 저장: {save_path}")
    
    return cleaned

def receipt_preprocessing_ultra_sharp(image_path, save_path=None):
    """초선명 영수증 전처리 (최고 품질)"""
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    print(f"원본 크기: {w} x {h}")
    
    # 1. 대폭 확대 (3배)
    scale = 3.0
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    print(f"3배 확대: {new_w} x {new_h}")
    
    # 2. 그레이스케일
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 3. 노이즈 제거 (edge-preserving)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 4. 히스토그램 평활화
    equalized = cv2.equalizeHist(denoised)
    
    # 5. 강력한 샤프닝
    kernel_sharp = np.array([[-1,-1,-1,-1,-1],
                            [-1, 2, 2, 2,-1],
                            [-1, 2, 8, 2,-1],
                            [-1, 2, 2, 2,-1],
                            [-1,-1,-1,-1,-1]]) / 8.0
    sharpened = cv2.filter2D(equalized, -1, kernel_sharp)
    
    # 6. 대비 증가
    contrasted = cv2.convertScaleAbs(sharpened, alpha=1.3, beta=10)
    
    # 7. 이진화 (여러 방법 시도 후 최적 선택)
    # 방법 1: Otsu
    _, otsu = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 방법 2: 적응적
    adaptive = cv2.adaptiveThreshold(contrasted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 15, 4)
    
    # 두 결과 비교 후 더 선명한 것 선택
    otsu_edges = cv2.Canny(otsu, 50, 150)
    adaptive_edges = cv2.Canny(adaptive, 50, 150)
    
    if np.sum(otsu_edges) > np.sum(adaptive_edges):
        binary = otsu
        print("Otsu 이진화 선택")
    else:
        binary = adaptive
        print("적응적 이진화 선택")
    
    # 8. 최종 정리
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    final = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    print(f"최종 크기: {final.shape[1]} x {final.shape[0]}")
    
    if save_path:
        cv2.imwrite(save_path, final)
        print(f"초선명 이미지 저장: {save_path}")
    
    return final

def receipt_preprocessing_balanced(image_path, save_path=None):
    """균형잡힌 영수증 전처리 (선명도 + 적당한 크기)"""
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    print(f"원본 크기: {w} x {h}")
    
    # 1. 적당한 확대 (2배)
    if h < 1200:
        scale = 2.0
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"2배 확대: {new_w} x {new_h}")
    else:
        resized = image
        print("크기 조정 없음")
    
    # 2. RGB to LAB 색공간 변환
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 3. L 채널에만 CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    
    # 4. LAB to BGR 변환
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 5. 그레이스케일
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    
    # 6. 미디언 필터 (노이즈 제거)
    filtered = cv2.medianBlur(gray, 3)
    
    # 7. 언샤프 마스킹
    gaussian = cv2.GaussianBlur(filtered, (5, 5), 1.0)
    unsharp = cv2.addWeighted(filtered, 1.5, gaussian, -0.5, 0)
    
    # 8. 적응적 이진화
    binary = cv2.adaptiveThreshold(unsharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    print(f"최종 크기: {binary.shape[1]} x {binary.shape[0]}")
    
    if save_path:
        cv2.imwrite(save_path, binary)
        print(f"균형잡힌 이미지 저장: {save_path}")
    
    return binary
def receipt_preprocessing_basic(image_path, save_path=None):
    """기본 영수증 전처리 파이프라인 (크기 보존)"""
    # 이미지 로드
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    
    print("기본 영수증 전처리 시작...")
    print(f"원본 크기: {original_width} x {original_height}")
    
    # 1. 이미지 크기 조정 (너무 크면 축소, 너무 작으면 확대)
    if original_height > 2000 or original_width > 1500:
        scale = min(2000/original_height, 1500/original_width)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"축소됨: {new_width} x {new_height}")
    elif original_height < 800:
        scale = min(2.0, 800 / original_height)  # 최대 2배까지만
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        print(f"확대됨: {new_width} x {new_height}")
    
    # 2. 가벼운 기울기 보정 (5도 이내만)
    gray_check = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_check, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is not None:
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            if angle < 45 or angle > 135:
                if angle > 135:
                    angle = angle - 180
                if abs(angle) < 5:  # 5도 이내만 보정
                    angles.append(angle)
        
        if angles:
            rotation_angle = np.median(angles)
            if abs(rotation_angle) > 0.5:
                h, w = image.shape[:2]
                center = (w//2, h//2)
                M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h), 
                                     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                print(f"기울기 보정: {rotation_angle:.2f}도")
    
    # 3. 대비 향상
    enhanced = enhance_receipt_contrast(image)
    
    # 4. 노이즈 제거
    denoised = remove_receipt_noise(enhanced)
    
    # 5. 보수적 여백 제거 (10% 이내만)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    _, binary_check = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary_check > 0))
    
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        h, w = gray.shape
        # 여백이 이미지의 10% 이상일 때만 제거
        if y_min > h * 0.1:
            y_min = max(0, y_min - int(h * 0.02))
        else:
            y_min = 0
            
        if (h - y_max) > h * 0.1:
            y_max = min(h, y_max + int(h * 0.02))
        else:
            y_max = h
            
        if x_min > w * 0.1:
            x_min = max(0, x_min - int(w * 0.02))
        else:
            x_min = 0
            
        if (w - x_max) > w * 0.1:
            x_max = min(w, x_max + int(w * 0.02))
        else:
            x_max = w
        
        denoised = denoised[y_min:y_max, x_min:x_max]
        print(f"여백 제거됨: {x_max-x_min} x {y_max-y_min}")
    
    # 6. 그레이스케일 변환
    final_gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    
    # 7. 최종 이진화
    binary = cv2.adaptiveThreshold(final_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    print(f"최종 크기: {binary.shape[1]} x {binary.shape[0]}")
    
    if save_path:
        cv2.imwrite(save_path, binary)
        print(f"처리된 이미지 저장: {save_path}")
    
    return binary

def receipt_preprocessing_advanced(image_path, save_path=None):
    """고급 영수증 전처리 파이프라인 (윤곽선 검출 포함)"""
    image = cv2.imread(image_path)
    
    print("고급 영수증 전처리 시작...")
    
    # 1. 영수증 윤곽선 검출 및 추출
    contour = detect_receipt_contour(image)
    
    if contour is not None:
        # 원근 변환으로 영수증 정렬
        warped = four_point_transform(image, contour.reshape(4, 2))
        print("영수증 윤곽선 검출 성공 - 원근 변환 적용")
    else:
        warped = image
        print("영수증 윤곽선 검출 실패 - 원본 이미지 사용")
    
    # 2. 크기 최적화
    h, w = warped.shape[:2]
    if h > 2000:
        scale = 2000 / h
        new_w = int(w * scale)
        warped = cv2.resize(warped, (new_w, 2000), interpolation=cv2.INTER_AREA)
    elif h < 1000:
        scale = 1000 / h
        new_w = int(w * scale)
        warped = cv2.resize(warped, (new_w, 1000), interpolation=cv2.INTER_CUBIC)
    
    # 3. 열전사 프린터 특화 처리
    enhanced = thermal_printer_enhancement(warped)
    
    # 4. 여백 제거
    cropped = crop_receipt_margins(enhanced)
    
    # 5. 최종 정리
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, kernel)
    
    if save_path:
        cv2.imwrite(save_path, cleaned)
        print(f"처리된 이미지 저장: {save_path}")
    
    return cleaned

def receipt_preprocessing_photo(image_path, save_path=None):
    """스마트폰으로 촬영한 영수증 전처리"""
    image = cv2.imread(image_path)
    
    print("스마트폰 촬영 영수증 전처리 시작...")
    
    # 1. 영수증 윤곽선 검출 (필수)
    contour = detect_receipt_contour(image)
    
    if contour is not None:
        # 원근 변환
        warped = four_point_transform(image, contour.reshape(4, 2))
        print("영수증 영역 추출 성공")
    else:
        # 윤곽선 검출 실패시 중앙 영역 크롭
        h, w = image.shape[:2]
        margin_h, margin_w = int(h * 0.1), int(w * 0.1)
        warped = image[margin_h:h-margin_h, margin_w:w-margin_w]
        print("윤곽선 검출 실패 - 중앙 영역 사용")
    
    # 2. 크기 조정 (영수증 비율 고려)
    h, w = warped.shape[:2]
    target_width = 800
    target_height = int(h * (target_width / w))
    resized = cv2.resize(warped, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    # 3. 조명 불균일 보정
    enhanced = enhance_receipt_contrast(resized)
    
    # 4. 노이즈 제거 (카메라 노이즈)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 5. 샤프닝
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    
    # 6. 적응적 이진화
    binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 15, 4)
    
    # 7. 최종 정리
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    if save_path:
        cv2.imwrite(save_path, cleaned)
        print(f"처리된 이미지 저장: {save_path}")
    
    return cleaned

def smart_receipt_preprocessing(image_path, receipt_type='auto', save_path=None):
    """
    스마트 영수증 전처리 (타입 자동 감지)
    
    Args:
        image_path: 입력 이미지 경로
        receipt_type: 'auto', 'scan', 'photo', 'thermal'
        save_path: 저장 경로
    """
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # 자동 타입 감지
    if receipt_type == 'auto':
        aspect_ratio = h / w
        
        # 가로세로 비율과 해상도로 타입 추정
        if aspect_ratio > 3 and w < 1000:  # 세로로 긴 저해상도
            receipt_type = 'thermal'
        elif h > 2000 or w > 1500:  # 고해상도
            receipt_type = 'scan'
        else:  # 일반 사진
            receipt_type = 'photo'
        
        print(f"자동 감지된 영수증 타입: {receipt_type}")
    
    # 타입별 전처리
    if receipt_type == 'scan':
        result = receipt_preprocessing_basic(image_path, save_path)
    elif receipt_type == 'photo':
        result = receipt_preprocessing_photo(image_path, save_path)
    elif receipt_type == 'thermal':
        image = cv2.imread(image_path)
        result = thermal_printer_enhancement(image)
        if save_path:
            cv2.imwrite(save_path, result)
    else:
        result = receipt_preprocessing_basic(image_path, save_path)
    
    return result

# 사용 예제
if __name__ == "__main__":
    # 예제 사용법
    image_path = "image/Recipt_clean.jpg"  # 실제 영수증 이미지 경로
    
    print("=== 영수증 전처리 테스트 ===\n")
    
    # 1. 선명도 우선 (권장)
    processed = receipt_preprocessing_sharp(image_path, save_path="image/receipt_sharp.jpg")
    
    # 2. 초선명 (최고 품질, 큰 파일)
    #ultra_sharp = receipt_preprocessing_ultra_sharp(image_path, "image/receipt_ultra.jpg")
    
    # 3. 균형잡힌 처리
    #balanced = receipt_preprocessing_balanced(image_path, "image/receipt_balanced.jpg")
    
    print("\n영수증 전처리 완료!")
    print("이제 OCR을 실행하세요:")
    print("""
    from PIL import Image
    import pytesseract
    
    img = Image.fromarray(processed)
    # 영수증 특화 OCR 설정
    config = '--psm 6 --oem 3'
    text = pytesseract.image_to_string(img, lang='kor+eng', config=config)
    print(text)
    """)

    img = Image.fromarray(processed)
    # 영수증 특화 OCR 설정
    config = '--psm 6 --oem 3'
    text = pytesseract.image_to_string(img, lang='kor+eng', config=config)
    print(text)