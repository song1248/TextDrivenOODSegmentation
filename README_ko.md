# TDOS: 텍스트 기반 의미 변이를 활용한 견고한 OOD 세분화

이 저장소는 **“Leveraging Text-Driven Semantic Variation for Robust OOD Segmentation”** 프로젝트와 함께 제공되며, 자율주행 시나리오에서 분포 외(OOD) 의미론적 세분화를 위해 텍스트를 활용하는 프레임워크를 소개합니다. 이 코드베이스는 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 생태계를 확장하여, CLIP 기반의 시각·텍스트 추론을 통해 이전에 관찰되지 않은 도로 이상 상황을 더욱 신뢰성 있게 탐지합니다.

## 개요

- **텍스트 기반 OOD 세분화** – CLIP 비전 트랜스포머 혹은 ResNet 백본과 CLIP 텍스트 인코더, Mask2Former 스타일 디코더를 결합하여 시각 특징과 유연한 텍스트 프롬프트를 정렬합니다.
- **거리 기반 OOD 프롬프트** – 분포 내(ID) 클래스에서 의미적 거리가 다른 프롬프트(논문에서는 WordNet 기반)를 생성해 ID/OOD 의사결정 경계를 선명하게 만듭니다.
- **OOD 의미 증강** – 외부 객체를 삽입하지 않고, 자체 어텐션 특징 교란을 이용해 다양한 OOD 프로토타입을 합성합니다.
- **비전/텍스트 정규화** – 사전 학습된 비전-언어 전문가를 고정한 채 프롬프트 파라미터와 디코더 헤드를 학습하며, 안정성을 위한 보조 아이덴티티 헤드를 포함합니다.
- **최첨단 OOD 세분화 성능** – Fishyscapes, Segment-Me-If-You-Can, Road Anomaly 벤치마크에서 픽셀 및 객체 기반 베이스라인을 능가합니다.

## 저장소 구성

- `configs/`: 실험 설정 파일. `configs/tdos/*.py`에서 CLIP ViT-B 및 ResNet 백본과 다양한 데이터셋을 다룹니다.
- `models/`: 사용자 정의 세그멘터(`TextDrivenOODSeg`)와 CLIP 백본, 유틸리티.
- `mmseg/`: 새로운 모델에 필요한 MMSegmentation 모듈(데이터셋, 러너, 레이어) 포크.
- `pretrained/`: CLIP 체크포인트 위치(예: `CLIP-ViT-B-16.pt`, `RN50.pt`).
- `tools/`: MMSegmentation에서 가져온 실행 유틸리티(데이터셋 변환기, 분산 실행 스크립트 등).
- `train.py`: 설정, 데이터셋, 로깅을 연결하는 싱글 노드 학습 엔트리 포인트.
- `dist_train.sh`: PyTorch 분산 학습을 위한 멀티 GPU 학습 보조 스크립트.

## 환경 설정

1. **Python 환경 생성** (CUDA 11.x + PyTorch ≥ 2.0 권장):
   ```bash
   conda create -n tdos python=3.9
   conda activate tdos
   ```
2. **CUDA 툴킷에 맞는 PyTorch 설치**:
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
   ```
3. **프로젝트 의존성 설치**:
   ```bash
   pip install -r requirements.txt
   ```
4. **CLIP 체크포인트 배치** (`pretrained/` 디렉터리):
   - ViT 백본용 `CLIP-ViT-B-16.pt`
   - ResNet 백본용 `RN50.pt`
   사용하려는 설정 파일의 `pretrained` 항목과 파일명이 일치해야 합니다.

## 데이터셋 준비

설정 파일은 기본적으로 Cityscapes를 ID 데이터셋으로 가정하고, 여러 OOD 벤치마크를 평가용으로 사용합니다.

- **Fishyscapes (Lost & Found / Static)**
- **Segment-Me-If-You-Can (Anomaly Track / Obstacle Track)**
- **Road Anomaly**

학습 전, 아래 기본 설정 파일의 데이터셋 경로를 환경에 맞게 수정하세요.

- `configs/_base_/datasets/city2city-512.py`
- `configs/_base_/datasets/city2fishy_*.py`
- `configs/_base_/datasets/city2road_anomaly*.py`

각 설정 파일에는 작성자의 환경(예: `/home/jovyan/...`)에 맞춘 `data_root`가 정의되어 있습니다. 이를 로컬 경로(`img_dir`, `ann_dir`)로 교체하세요.

## 학습

### 싱글 노드 / 싱글 GPU

```bash
python train.py configs/tdos/text_driven_ood_seg_vit-b_1e-5_20k-c2c-512.py \
    --work-dir ./work_dirs/tdos_vit_b_cityscapes
```

- 로그와 체크포인트는 `work_dirs/<config_name>/`에 저장됩니다.
- 사전 학습된 체크포인트로부터 시작하려면 `--load-from`, 학습 재개 시 `--resume-from`을 사용하세요.
- `--finetune` 옵션은 학습 전에 모델을 평가 모드로 전환하여 디코더/프롬프트 헤드만 학습합니다.

### 멀티 GPU (분산 학습)

```bash
bash dist_train.sh configs/tdos/text_driven_ood_seg_vit-b_1e-5_20k-c2fishy_LnF-512.py 4 \
    --work-dir ./work_dirs/tdos_vit_b_fishyscapes
```

`dist_train.sh`는 전달된 추가 인자를 `train.py`에 그대로 넘깁니다. 특정 GPU만 사용하려면 실행 전에 `CUDA_VISIBLE_DEVICES`를 설정하세요.

### 하이퍼파라미터

- 옵티마이저: `AdamW`, 모듈별 학습률 배수 사용(기본적으로 백본/텍스트 인코더는 고정).
- 프롬프트 길이: 설정 파일의 `context_length`로 제어하며, `TextDrivenOODSeg`에서 동적으로 확장됩니다.
- `learnable_cls_num` 기본값 50은 디코더 쿼리에 추가되는 네거티브 프롬프트 슬롯 수를 결정합니다.

새 데이터셋에 맞게 설정 파일 또는 세그멘터 내부를 조정하세요.

## 평가 및 추론

체크포인트 평가는 대응되는 설정 파일과 함께 실행합니다.

```bash
bash dist_test.sh configs/tdos/text_driven_ood_seg_vit-b_1e-5_20k-c2road_anomaly-512.py \
    PRETRAINED_CHECKPOINT.pth 4 --eval mIoU
```

결과 시각화가 필요하다면 `tools/test.py`에 `--show-dir`을 추가하거나 `train.py`에 맞춤 훅을 넣어 활용하십시오.

## 인용

이 연구를 기반으로 작업한다면 다음을 인용해 주세요.

```
@inproceedings{song2024tdos,
  title     = {Leveraging Text-Driven Semantic Variation for Robust OOD Segmentation},
  author    = {Seungheon Song and Jaekoo Lee},
  booktitle = {Proceedings of the Kookmin University Autonomous Systems Symposium},
  year      = {2024}
}
```

## 감사의 말

본 연구는 [CLIP](https://github.com/openai/CLIP)과 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)을 기반으로 구축되었습니다. 두 프로젝트의 라이선스를 확인하고, 연구에 활용할 경우 적절히 인용해 주시기 바랍니다.
