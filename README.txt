# Netlify 배포용 웹 앱 템플릿

이것은 모델 파일을 제외한 웹 앱의 템플릿입니다. `Train_Fault_Classifier.ipynb` 노트북을 실행하여 얻은 3개의 `.pkl` 파일을 이 폴더에 추가한 후 배포를 진행하세요.

## 🚀 배포 방법

1.  **모델 파일 준비**: Google Colab에서 `Train_Fault_Classifier.ipynb`를 실행하여 `model.pkl`, `scaler.pkl`, `label_encoder.pkl` 파일을 다운로드합니다.
2.  **파일 합치기**: 다운로드한 3개의 `.pkl` 파일을 이 폴더 안에 넣습니다.
3.  **GitHub에 업로드**: 모든 파일(총 8개)이 포함된 이 폴더를 GitHub 저장소에 업로드합니다.
4.  **Netlify에 배포**: GitHub 저장소를 Netlify에 연결하여 배포를 시작합니다.