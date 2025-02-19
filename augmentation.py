# albumentations 설치 필요
from itertools import combinations
from pathlib import Path
import albumentations as A
import cv2
import numpy as np


# annotation한 라벨 로드
def load_label(label_path):
    labels = np.loadtxt(label_path, delimiter=' ') if label_path.exists() else np.array([])
    # 커스텀데이터셋의 객체는 다 1개씩이므로 라벨은 모두 1줄이라 shape이 (5,)인 1차원 배열임
    # albumentations의 bboxparams로 인자를 넣어주기 위해서,
    # shape이 (5,)인 1차원 배열을 (1, 5)인 2차원 배열로 맞춰줌
    return labels.reshape(-1, 5) if labels.size > 0 else labels

def augment_dataset(data_type, transform, aug_idx):
    base_path = Path('datasets/custom_data')
    src_img_path = base_path / data_type / 'images'
    src_label_path = base_path / data_type / 'labels'
    dst_img_path = base_path / 'augmented' / data_type / 'images'
    dst_label_path = base_path / 'augmented' / data_type / 'labels'
    
    dst_img_path.mkdir(parents=True, exist_ok=True)
    dst_label_path.mkdir(parents=True, exist_ok=True)

    for img_path in src_img_path.glob('*.jpg'):
        name = img_path.stem
        image = cv2.imread(str(img_path))
        # 용량 압축을 위한 리사이즈
        # (yolo의 라벨은 0~1사이의 정규화된 값으로 상대적 비율임. 리사이즈해도 bbox수정할 필욘 없음)
        h, w = image.shape[:2]
        r = 640 / max(h, w)
        new_w, new_h = int(w * r), int(h * r)
        image = cv2.resize(image, (new_w, new_h))
        labels = load_label(src_label_path / f'{name}.txt')
        
        if len(labels) > 0:
            bboxes = labels[:, 1:]
            class_labels = labels[:, 0]

            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

            # 공간적 정보가 변하는 spatial-level transform을 넣어놨으므로,
            # 증강 이후의 bbox정보를 가져오고 yolo의 라벨형식으로 합쳐주도록함
            if len(augmented['bboxes']) > 0:
                aug_bboxes = np.array(augmented['bboxes'])
                aug_labels = np.column_stack([augmented['class_labels'], aug_bboxes])
            else:
                aug_labels = np.array([])
            
            cv2.imwrite(str(dst_img_path / f'aug_{aug_idx}_{name}.jpg'), augmented['image'])
            # savetxt함수 정의를 보면 기본포맷이 %.18e이므로, 
            # yolo의 라벨에 맞게 클래스id는 int, bbox정보는 float으로 저장해줌
            np.savetxt(dst_label_path / f'aug_{aug_idx}_{name}.txt', aug_labels, fmt='%d %.6f %.6f %.6f %.6f')

def main():
    # 카메라로 촬영한 데이터셋이므로, 카메라 플래시나 노이즈등에 대응할만한 transform들 생성
    augmentations = [
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Rotate(limit=5, p=0.5),
        A.Sharpen(alpha=(0.2, 0.3), lightness=(0.7, 1.0), p=0.5),
        A.RandomShadow(num_shadows_limit=(1,2), shadow_dimension=5, p=0.5),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.5),
    ]

    # SomeOf를 이용하여 매번 랜덤으로 2개의 증강을 적용하게 함(호출마다 랜덤)
    # 반복으로 같은 조합이 나올 수도 있지만, transform들의 수치가 랜덤이라 완전히 같지는 않음
    # 원본 데이터셋의 이미지는 50장이므로, 총 50 + 50*6 = 350장이 만들어짐
    transform = A.Compose([
        A.SomeOf(augmentations, n=2, p=0.7, replace=False),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    for data_type in ['train', 'valid', 'test']:
        print(f'-증강중인 데이터셋: {data_type}...')
        for i in range(6):
            # 범위 만큼 반복해서 증강 적용
            augment_dataset(data_type, transform, aug_idx=i)
        print(f'-완료된 데이터셋: {data_type}')

if __name__ == '__main__':
    main()
