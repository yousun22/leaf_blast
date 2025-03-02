import os  # os 모듈 추가
import requests
import xml.etree.ElementTree as ET
import cv2
import random
import numpy as np
from io import BytesIO


class BlastSlice:
    def __init__(self, image_base_url, annotation_url, window_size=256, training_range=(0, 159)):
        self.image_base_url = image_base_url  # GitHub Raw 이미지 URL
        self.annotation_file = "all_annotations.xml"  # XML 파일명 고정
        self.window_size = window_size
        self.training_range = training_range

        # GitHub에서 XML 파일 다운로드
        response = requests.get(annotation_url)
        if response.status_code == 200:
            with open(self.annotation_file, "w") as f:
                f.write(response.text)
            print("📂 all_annotations.xml 다운로드 완료!")
        else:
            raise Exception("❌ XML 파일을 불러올 수 없습니다.")

        self.annotations = self.parse_annotation()

    def parse_annotation(self):
        """ XML 어노테이션 파일을 파싱하여 이미지별 바운딩 박스 정보를 저장 """
        tree = ET.parse(self.annotation_file)
        root = tree.getroot()
        annotations = {}

        for image in root.findall('image'):
            image_name = image.get('name').strip().lower()
            boxes = []
            for box in image.findall('box'):
                xmin = float(box.get('xtl'))
                ymin = float(box.get('ytl'))
                xmax = float(box.get('xbr'))
                ymax = float(box.get('ybr'))
                if xmin >= 0 and ymin >= 0 and xmax > xmin and ymax > ymin:
                    boxes.append((xmin, ymin, xmax, ymax))
            annotations[image_name] = boxes
        return annotations

    def slice(self, window_size=None):
        """ GitHub에서 직접 이미지를 불러와 바운딩 박스를 기준으로 슬라이싱하고 반환 """
        if window_size:
            self.window_size = window_size

        extracted_images = {}  # {파일명: 이미지 배열} 형태로 반환
        extracted_annotations = {}  # {파일명: 어노테이션 내용} 형태로 반환

        for image_name, boxes in self.annotations.items():
            image_number = int(image_name.split(".")[0])
            if image_number > self.training_range[1]:
                continue  # 테스트 이미지는 건너뜀

            # GitHub Raw URL 생성
            image_url = f"{self.image_base_url}/{image_name}"

            # GitHub에서 직접 이미지 요청
            response = requests.get(image_url)
            if response.status_code != 200:
                print(f"❌ Error: {image_name} could not be loaded.")
                continue

            # OpenCV로 이미지 디코딩
            img_arr = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if image is None:
                print(f"❌ Error: Failed to decode {image_name}")
                continue

            img_height, img_width = image.shape[:2]
            image_counter = 1

            for (xmin, ymin, xmax, ymax) in boxes:
                box_width = xmax - xmin
                box_height = ymax - ymin

                if box_width > self.window_size or box_height > self.window_size:
                    print(f"Box too large for {image_name}, skipping.")
                    continue

                x_random_offset = random.randint(-int((self.window_size - box_width) / 2), int((self.window_size - box_width) / 2))
                y_random_offset = random.randint(-int((self.window_size - box_height) / 2), int((self.window_size - box_height) / 2))

                x_start = max(0, int(xmin) - x_random_offset)
                y_start = max(0, int(ymin) - y_random_offset)
                x_end = min(img_width, x_start + self.window_size)
                y_end = min(img_height, y_start + self.window_size)

                cropped_img = image[y_start:y_end, x_start:x_end]

                # 이미지가 window_size보다 작으면 검은색으로 채움
                top_padding = max(0, self.window_size - (y_end - y_start))
                left_padding = max(0, self.window_size - (x_end - x_start))

                cropped_img_padded = cv2.copyMakeBorder(
                    cropped_img,
                    top=top_padding,
                    bottom=0,
                    left=left_padding,
                    right=0,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )

                boxes_in_window = [
                    box for box in boxes if box[0] >= x_start and box[1] >= y_start and box[2] <= x_end and box[3] <= y_end
                ]
                if not boxes_in_window:
                    continue

                cropped_image_name = f"{os.path.splitext(image_name)[0]}-{image_counter}.jpg"
                annotation_data = []

                for box in boxes_in_window:
                    xmin, ymin, xmax, ymax = box
                    x_center = (xmin + xmax) / 2 - x_start + left_padding
                    y_center = (ymin + ymax) / 2 - y_start + top_padding
                    bbox_width = xmax - xmin
                    bbox_height = ymax - ymin
                    annotation_data.append(f"0 {x_center / self.window_size} {y_center / self.window_size} {bbox_width / self.window_size} {bbox_height / self.window_size}")

                # 이미지 저장 없이 메모리에 저장 (BytesIO 이용)
                _, encoded_img = cv2.imencode(".jpg", cropped_img_padded)
                img_buffer = BytesIO(encoded_img.tobytes())

                extracted_images[cropped_image_name] = img_buffer
                extracted_annotations[cropped_image_name.replace(".jpg", ".txt")] = "\n".join(annotation_data)

                image_counter += 1

        return extracted_images, extracted_annotations
