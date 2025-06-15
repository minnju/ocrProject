import cv2
import numpy as np
import easyocr

def aggressive_preprocess_receipt(image_path):
    """
    초강력 영수증 전처리 - 모든 기법 동원
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return None
    
    print("1. 초고해상도 변환 시작...")
    # 1. 해상도 5배 증가 (더 극단적으로)
    height, width = img.shape[:2]
    img_huge = cv2.resize(img, (width * 5, height * 5), interpolation=cv2.INTER_CUBIC)
    
    # 2. 그레이스케일
    gray = cv2.cvtColor(img_huge, cv2.COLOR_BGR2GRAY)
    
    print("2. 강력한 노이즈 제거...")
    # 3. Bilateral 필터 (강력한 노이즈 제거하면서 경계 보존)
    bilateral = cv2.bilateralFilter(gray, 15, 80, 80)
    
    print("3. 극단적 대비 향상...")
    # 4. 극단적 CLAHE (대비 최대 강화)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(bilateral)
    
    print("4. 초강력 샤프닝...")
    # 5. 매우 강한 샤프닝
    kernel_sharpen = np.array([[-3, -3, -3],
                              [-3, 25, -3],
                              [-3, -3, -3]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
    
    print("5. 감마 보정...")
    # 6. 감마 보정 (밝기 조절)
    gamma = 1.2
    gamma_corrected = np.power(sharpened / 255.0, gamma) * 255.0
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    print("6. 적응형 이진화...")
    # 7. 여러 방법의 이진화 시도
    # 방법 1: OTSU
    _, otsu = cv2.threshold(gamma_corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 방법 2: 적응형 (가우시안)
    adaptive_gauss = cv2.adaptiveThreshold(gamma_corrected, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 
                                         blockSize=21, C=10)
    
    # 방법 3: 적응형 (평균)
    adaptive_mean = cv2.adaptiveThreshold(gamma_corrected, 255, 
                                        cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 
                                        blockSize=21, C=10)
    
    print("7. 최적 이진화 선택...")
    # 가장 좋은 이진화 방법 선택 (여기서는 적응형 가우시안)
    binary = adaptive_gauss
    
    print("8. 모폴로지 연산...")
    # 8. 텍스트 연결 개선
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # 9. 작은 노이즈 제거
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    final = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    print("전처리 완료!")
    return final

def try_multiple_preprocessing_methods(image_path):
    """
    여러 전처리 방법을 시도하고 가장 좋은 결과 반환
    """
    methods = []
    
    # 방법 1: 기본 전처리
    def method1(img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    # 방법 2: 강화된 전처리
    def method2(img_path):
        return aggressive_preprocess_receipt(img_path)
    
    # 방법 3: 다른 접근
    def method3(img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 매우 큰 크기로 확대
        resized = cv2.resize(gray, None, fx=6, fy=6, interpolation=cv2.INTER_LANCZOS4)
        
        # 언샤프 마스킹
        gaussian = cv2.GaussianBlur(resized, (9, 9), 2.0)
        unsharp = cv2.addWeighted(resized, 2.0, gaussian, -1.0, 0)
        
        # 히스토그램 평활화
        equalized = cv2.equalizeHist(unsharp)
        
        # 적응형 이진화
        binary = cv2.adaptiveThreshold(equalized, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 31, 15)
        return binary
    
    methods = [
        ("기본 OTSU 방법", method1),
        ("강화된 전처리", method2), 
        ("언샤프 마스킹", method3)
    ]
    
    return methods

def ultimate_receipt_ocr(image_path):
    """
    모든 전처리 방법을 시도하여 최고 결과 찾기
    """
    reader = easyocr.Reader(['ko', 'en'], gpu=False)
    methods = try_multiple_preprocessing_methods(image_path)
    
    best_result = None
    best_score = 0
    best_method_name = ""
    
    for method_name, method_func in methods:
        print(f"\n=== {method_name} 시도 중... ===")
        
        try:
            # 전처리 실행
            processed_img = method_func(image_path)
            if processed_img is None:
                continue
                
            # 전처리된 이미지 저장
            processed_path = f'image/processed_{method_name.replace(" ", "_")}.png'
            cv2.imwrite(processed_path, processed_img)
            
            # 여러 OCR 파라미터 세트 시도
            ocr_configs = [
                # 설정 1: 기본
                {
                    'width_ths': 0.7, 'height_ths': 0.7, 'text_threshold': 0.7,
                    'low_text': 0.4, 'link_threshold': 0.4, 'canvas_size': 3840,
                    'mag_ratio': 1.5, 'min_size': 10
                },
                # 설정 2: 관대한 설정
                {
                    'width_ths': 0.5, 'height_ths': 0.5, 'text_threshold': 0.5,
                    'low_text': 0.3, 'link_threshold': 0.3, 'canvas_size': 5120,
                    'mag_ratio': 2.0, 'min_size': 8
                },
                # 설정 3: 엄격한 설정  
                {
                    'width_ths': 0.9, 'height_ths': 0.9, 'text_threshold': 0.9,
                    'low_text': 0.6, 'link_threshold': 0.6, 'canvas_size': 4096,
                    'mag_ratio': 1.8, 'min_size': 15
                }
            ]
            
            for i, config in enumerate(ocr_configs):
                print(f"  OCR 설정 {i+1} 시도...")
                
                result = reader.readtext(processed_path,
                                       detail=1,
                                       paragraph=False,
                                       decoder='beamsearch',
                                       beamWidth=5,
                                       batch_size=1,
                                       **config)
                
                if result:
                    # 결과 품질 평가
                    total_confidence = sum(conf for _, _, conf in result)
                    text_count = len([text for _, text, conf in result if len(text.strip()) > 1])
                    avg_confidence = total_confidence / len(result) if result else 0
                    
                    # 점수 계산 (텍스트 개수 * 평균 신뢰도)
                    score = text_count * avg_confidence
                    
                    print(f"    텍스트 개수: {text_count}, 평균 신뢰도: {avg_confidence:.2f}, 점수: {score:.2f}")
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_method_name = f"{method_name} (설정 {i+1})"
                        
        except Exception as e:
            print(f"  오류 발생: {e}")
            continue
    
    print(f"\n🏆 최고 결과: {best_method_name} (점수: {best_score:.2f})")
    return best_result

def filter_and_format_receipt(ocr_result, min_confidence=0.3):
    """
    영수증 OCR 결과 필터링 및 포맷팅
    """
    if not ocr_result:
        return []
    
    # 신뢰도 필터링
    filtered_results = []
    for bbox, text, confidence in ocr_result:
        if confidence >= min_confidence:
            # 텍스트 정리
            cleaned_text = text.strip()
            if len(cleaned_text) > 1:  # 한 글자는 제외
                filtered_results.append((bbox, cleaned_text, confidence))
    
    # y 좌표로 정렬 (위에서 아래로)
    filtered_results.sort(key=lambda x: x[0][0][1])
    
    return filtered_results

def main():
    """
    메인 실행 함수 - 최강 전처리 버전
    """
    image_path = 'recipt_example.jpg'  # 실제 이미지 경로로 변경
    
    print("🔥 최강 영수증 OCR 처리 시작! 🔥")
    print("여러 전처리 방법과 OCR 설정을 모두 시도합니다...")
    
    # 최고의 OCR 결과 찾기
    best_result = ultimate_receipt_ocr(image_path)
    
    if best_result:
        # 결과 필터링 및 출력
        filtered_results = filter_and_format_receipt(best_result, min_confidence=0.2)  # 더 관대하게
        
        print(f"\n🎯 최종 OCR 결과 (총 {len(filtered_results)}개 텍스트)")
        print("=" * 80)
        
        for i, (bbox, text, confidence) in enumerate(filtered_results):
            print(f"{i+1:2d}. 텍스트: {text:<40} 신뢰도: {confidence:.2f}")
        
        # 신뢰도별 분류
        high_conf = [r for r in filtered_results if r[2] >= 0.7]
        medium_conf = [r for r in filtered_results if 0.4 <= r[2] < 0.7]
        low_conf = [r for r in filtered_results if 0.2 <= r[2] < 0.4]
        
        print(f"\n📊 신뢰도 분석:")
        print(f"   🟢 높음 (0.7+):   {len(high_conf)}개")
        print(f"   🟡 보통 (0.4-0.7): {len(medium_conf)}개") 
        print(f"   🔴 낮음 (0.2-0.4): {len(low_conf)}개")
        
        if high_conf:
            print(f"\n🔥 고신뢰도 텍스트:")
            for bbox, text, confidence in high_conf:
                print(f"   ✅ {text} (신뢰도: {confidence:.2f})")
                
        # 추가 팁 제공
        if len(filtered_results) < 10:
            print(f"\n💡 개선 팁:")
            print(f"   - 이미지를 더 밝은 곳에서 촬영해보세요")
            print(f"   - 카메라를 영수증에 더 가까이 대보세요") 
            print(f"   - 영수증을 평평하게 펴서 촬영해보세요")
            print(f"   - 그림자가 없는 곳에서 촬영해보세요")
    
    else:
        print("😱 모든 방법이 실패했습니다!")
        print("💡 다음을 시도해보세요:")
        print("   1. 이미지를 스캔하거나 더 고화질로 촬영")
        print("   2. 조명을 개선하여 재촬영") 
        print("   3. 다른 OCR 서비스 (Google Cloud Vision API) 시도")

# 디버깅용 함수 추가
def debug_preprocessing(image_path):
    """
    전처리 과정을 단계별로 저장하여 확인
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    
    # 각 단계별 저장
    cv2.imwrite('debug_01_original.png', img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('debug_02_gray.png', gray)
    
    resized = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('debug_03_resized.png', resized)
    
    bilateral = cv2.bilateralFilter(resized, 15, 80, 80)
    cv2.imwrite('debug_04_bilateral.png', bilateral)
    
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(bilateral)
    cv2.imwrite('debug_05_clahe.png', enhanced)
    
    kernel_sharpen = np.array([[-3, -3, -3], [-3, 25, -3], [-3, -3, -3]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
    cv2.imwrite('debug_06_sharpened.png', sharpened)
    
    binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 21, 10)
    cv2.imwrite('debug_07_binary.png', binary)
    
    print("디버그 이미지들이 저장되었습니다. 각 단계를 확인해보세요!")

if __name__ == "__main__":
    main()