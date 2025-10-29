# 🛰️ GPR Cavity Segmentation
> GPR(지표투과레이더) 데이터를 이용해 지하 공동(cavity) 영역을 자동으로 분할·시각화하는 딥러닝 모델

---

## 📖 프로젝트 개요 (Overview)
지표투과레이더(Ground Penetrating Radar, GPR)는 지하 구조나 공동을 탐지하는 데 널리 사용되는 기술입니다.  
그러나 GPR 데이터를 전문가가 수작업으로 분석하는 과정은 시간이 많이 들고, 주관적 판단에 의존합니다.  

이 프로젝트는 **GPR B-scan 이미지를 입력으로 받아**,  
**딥러닝 기반 세그멘테이션(U-Net) 모델을 통해 자동으로 공동 마스크를 생성하는 시스템**을 구현하는 것을 목표로 합니다.

---

## 🚀 주요 기능 (Features)
- 🧠 GPR 이미지–공동 마스크 쌍 데이터를 이용한 **지도학습(Supervised Learning)**  
- 🧩 **U-Net 기반 세그멘테이션 모델** (PyTorch 구현)  
- 🧪 **다중 단면 융합(2.5D 구조)** 으로 확장 가능  
- 🖼️ 예측 결과를 GPR 원본 이미지 위에 **시각적으로 오버레이**  
- ⚙️ **확장 가능한 데이터 구조** (지역별 site 단위 관리)

---

## 📂 폴더 구조 (Project Structure)

<pre><code>
GPR_Cavity_Segmentation/
├── data/
│   ├── site_001/
│   │   ├── images/
│   │   │   ├── 001_1.jpg
│   │   │   ├── 001_2.jpg
│   │   └── masks/
│   │       ├── 001_1_mask.png
│   │       └── 001_2_mask.png
│   └── site_002/
│       └── ...
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
│
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   └── logs/
│
└── README.md
</code></pre>

---

## 🧠 모델 구조 (Model Architecture)
기본 모델은 **U-Net (ResNet-34 인코더, ImageNet 사전학습 가중치)** 구조를 기반으로 합니다.  
픽셀 단위로 공동 영역을 분할(segmentation)하며, 손실함수는 Dice + BCE 조합을 사용합니다.

