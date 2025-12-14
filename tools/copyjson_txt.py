import os
import json
from pathlib import Path

def convert_bbox(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = map(float, bbox)
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height
    return x_center, y_center, w, h

def json_to_yolo(json_path, output_path):
    # 빈 JSON은 건너뛰기
    if os.stat(json_path).st_size == 0:
        print(f"❌ Empty JSON: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    width  = int(data['FileInfo']['Width'])
    height = int(data['FileInfo']['Height'])
    bboxes = data['ObjectInfo']['BoundingBox']

    result_lines = []

    # ✅ Leye
    if bboxes['Leye']['isVisible']:
        class_id = 0 if bboxes['Leye']['Opened'] else 1
        coords = convert_bbox(bboxes['Leye']['Position'], width, height)
        result_lines.append(f"{class_id} {' '.join(map(str, coords))}")

    # ✅ Reye
    if bboxes['Reye']['isVisible']:
        class_id = 2 if bboxes['Reye']['Opened'] else 3
        coords = convert_bbox(bboxes['Reye']['Position'], width, height)
        result_lines.append(f"{class_id} {' '.join(map(str, coords))}")

    # ✅ Mouth
    if bboxes['Mouth']['isVisible']:
        class_id = 4 if bboxes['Mouth']['Opened'] else 5
        coords = convert_bbox(bboxes['Mouth']['Position'], width, height)
        result_lines.append(f"{class_id} {' '.join(map(str, coords))}")

    # ⚠️ Face는 제외 (6,7번 클래스 제거)

    # 저장
    os.makedirs(output_path, exist_ok=True)
    base_name = Path(json_path).stem
    out_file  = os.path.join(output_path, base_name + '.txt')
    with open(out_file, 'w') as f:
        f.write('\n'.join(result_lines))


def convert_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.json'):
                json_to_yolo(os.path.join(root, filename), output_dir)


# ===== 실제 실행 경로 =====
# 기존 txt 덮어쓰기 전 모두 비우기
# os.system("find '/home/jovyan/lost+found/ice_sleep_detpj/data/YOLO_dataset/labels/train' -type f -delete")
# os.system("find '/home/jovyan/lost+found/ice_sleep_detpj/data/YOLO_dataset/labels/test_data(통제)' -type f -delete")
# 1️⃣ Training JSON → YOLO txt
# convert_folder(
#     '/home/jovyan/lost+found/ice_sleep_detpj/ice_mobilenet_data/data/Training/label/bbox(실제도로환경)/승용',
#     '/home/jovyan/lost+found/ice_sleep_detpj/data/YOLO_dataset/labels/train'
# )

# 2️⃣ Validation JSON → YOLO txt
convert_folder(
    '/home/jovyan/lost+found/ice_sleep_detpj/ice_mobilenet_data/data/Validation/label/트럭',
    '/home/jovyan/lost+found/ice_sleep_detpj/data/YOLO_dataset/labels/test_data(트럭)'
)

print("✅ Train/Val 라벨 변환 완료!")
