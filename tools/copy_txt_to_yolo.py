import os
import json

def convert_bbox(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = map(float, bbox)
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height
    return x_center, y_center, w, h

def json_to_yolo_with_head(json_path, output_path, head_threshold=0.70):
    # ✅ 빈 JSON 파일 스킵
    if os.stat(json_path).st_size == 0:
        print(f"❌ Skipping empty JSON file: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    width = int(data['FileInfo']['Width'])
    height = int(data['FileInfo']['Height'])
    bboxes = data['ObjectInfo']['BoundingBox']

    result_lines = []

    if bboxes['Leye']['isVisible']:
        class_id = 0 if bboxes['Leye']['Opened'] else 1
        coords = convert_bbox(bboxes['Leye']['Position'], width, height)
        result_lines.append(f"{class_id} {' '.join(map(str, coords))}")

    if bboxes['Reye']['isVisible']:
        class_id = 2 if bboxes['Reye']['Opened'] else 3
        coords = convert_bbox(bboxes['Reye']['Position'], width, height)
        result_lines.append(f"{class_id} {' '.join(map(str, coords))}")

    if bboxes['Mouth']['isVisible']:
        class_id = 4 if bboxes['Mouth']['Opened'] else 5
        coords = convert_bbox(bboxes['Mouth']['Position'], width, height)
        result_lines.append(f"{class_id} {' '.join(map(str, coords))}")

    if bboxes['Face']['isVisible']:
        x_min, y_min, x_max, y_max = map(float, bboxes['Face']['Position'])
        face_center_y = (y_min + y_max) / 2
        face_center_ratio = face_center_y / height

        class_id = 7 if face_center_ratio > head_threshold else 6
        coords = convert_bbox(bboxes['Face']['Position'], width, height)
        result_lines.append(f"{class_id} {' '.join(map(str, coords))}")

    # 저장
    os.makedirs(output_path, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_file = os.path.join(output_path, base_name + '.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(result_lines))

def convert_json_folder(input_dir, output_dir, head_threshold=0.70):
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.json'):
                json_path = os.path.join(root, filename)
                json_to_yolo_with_head(json_path, output_dir, head_threshold=head_threshold)

# ========================
# ✅ 실제 실행
# ========================

# # 1. Training JSON → YOLO txt
# convert_json_folder(
#     input_dir='/home/jovyan/lost+found/ice_sleep_detpj/ice_mobilenet_data/data/Training/label/bbox(실제도로환경)/승용',
#     output_dir='/home/jovyan/lost+found/ice_sleep_detpj/data/YOLO_dataset/labels/train',
#     head_threshold=0.70
# )

# 2. Validation JSON → YOLO txt
convert_json_folder(
    input_dir='/home/jovyan/lost+found/ice_sleep_detpj/ice_mobilenet_data/data/Validation/label/승용',
    output_dir='/home/jovyan/lost+found/ice_sleep_detpj/data/YOLO_dataset/labels/val'
)
