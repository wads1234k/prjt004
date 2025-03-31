# Teachable Machine Webcam Object Recognition

이 프로젝트는 **Teachable Machine**과 **웹캠**을 이용하여 다양한 사물을 실시간으로 인식하는 프로그램입니다. 
사용자가 **Teachable Machine**을 통해 학습한 모델을 웹캠에서 실시간으로 예측하는 기능을 제공합니다.

키보드, 마우스, 컵 을 학습하여 이를 웹캠을 통해 실시간으로 처리합니다.


## 📦 프로젝트 구성

이 프로젝트는 **Python**과 **TensorFlow**를 사용하여 구축되었습니다. OpenCV를 사용하여 웹캠에서 이미지를 캡처하고,
Teachable Machine에서 학습한 모델을 이용해 객체를 실시간으로 인식합니다.

## 🛠️ 설치 및 환경 설정

### 1. 필수 라이브러리 설치

이 프로젝트를 실행하기 위해서는 다음의 Python 라이브러리를 설치해야 합니다.

```bash
pip install opencv-python tensorflow numpy
