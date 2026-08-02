"""
extract_ppocr.py   PaddleOCR  poster 

ppOCR ""
 extract_mineru.py  pipeline 
"""

import cv2
import numpy as np
from typing import List, Dict


class PPOCRExtractor:
    """PaddleOCR """

    def __init__(self, lang: str = "en", use_angle_cls: bool = True,
                 det_db_thresh: float = 0.3, min_image_area_ratio: float = 0.005):
        """
        Args:
            lang: OCR 'en' , 'ch' 
            use_angle_cls: 
            det_db_thresh: 
            min_image_area_ratio: 
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.det_db_thresh = det_db_thresh
        self.min_image_area_ratio = min_image_area_ratio
        self._ocr = None  # 

    def _get_ocr(self):
        """ PaddleOCR"""
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                use_gpu=False,   #  CPU cudnn 
                show_log=False,
            )
        return self._ocr

    def extract(self, image_path: str) -> List[Dict]:
        """
         poster 

         MinerUExtractor.extract() 
        - id: int, 
        - type: 'text' | 'image'
        - bbox: [x1, y1, x2, y2],  [0, 1000]
        - content: str, 
        - page_idx: 0
        """
        print("[ppOCR] ...")
        ocr = self._get_ocr()

        # 
        img_data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f": {image_path}")

        h, w = img.shape[:2]

        # OCR 
        results = ocr.ocr(image_path, cls=self.use_angle_cls)

        blocks = []
        text_mask = np.zeros((h, w), dtype=np.uint8)  # 

        if results and results[0]:
            for line in results[0]:
                points, (text, conf) = line

                # 
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                px1 = int(min(x_coords))
                py1 = int(min(y_coords))
                px2 = int(max(x_coords))
                py2 = int(max(y_coords))

                #  [0, 1000]
                bbox = [
                    px1 / w * 1000,
                    py1 / h * 1000,
                    px2 / w * 1000,
                    py2 / h * 1000,
                ]

                blocks.append({
                    "id": len(blocks),
                    "type": "text",
                    "bbox": bbox,
                    "content": text,
                    "confidence": conf,
                    "page_idx": 0,
                })

                # 
                cv2.rectangle(text_mask, (px1, py1), (px2, py2), 255, -1)

        print(f"[ppOCR]  {len(blocks)} ")

        # 
        image_blocks = self._detect_image_regions(img, text_mask, w, h)
        for ib in image_blocks:
            ib["id"] = len(blocks)
            blocks.append(ib)

        if image_blocks:
            print(f"[ppOCR]  {len(image_blocks)} ")

        print(f"[ppOCR]  {len(blocks)} ")
        return blocks

    def _pre_merge_lines(self, text_blocks: list, w: int, h: int) -> list:
        """
        
        :
          -  block 
          - 
         [0, 1000]1000  = /
        """
        if not text_blocks:
            return text_blocks

        # 
        text_blocks = sorted(text_blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))

        merged = []
        used = [False] * len(text_blocks)

        for i, b in enumerate(text_blocks):
            if used[i]:
                continue
            current = {
                "type": "text",
                "bbox": list(b["bbox"]),
                "content": b.get("content", ""),
                "page_idx": 0,
            }
            current_h = b["bbox"][3] - b["bbox"][1]
            used[i] = True

            #  block
            changed = True
            while changed:
                changed = False
                for j, c in enumerate(text_blocks):
                    if used[j]:
                        continue
                    cx1, cy1, cx2, cy2 = c["bbox"]
                    x1, y1, x2, y2 = current["bbox"]

                    # 
                    h_overlap = max(0, min(x2, cx2) - max(x1, cx1))
                    h_min_width = min(x2 - x1, cx2 - cx1)
                    if h_min_width <= 0:
                        continue
                    h_overlap_ratio = h_overlap / h_min_width

                    # cy1 - y2  c  current 
                    v_gap = cy1 - y2

                    line_h = max(current_h, cy2 - cy1)
                    #  > 50% < 0.7 
                    if h_overlap_ratio > 0.5 and -line_h < v_gap < line_h * 0.7:
                        current["bbox"] = [
                            min(x1, cx1), min(y1, cy1),
                            max(x2, cx2), max(y2, cy2),
                        ]
                        if c.get("content"):
                            current["content"] += " " + c["content"]
                        used[j] = True
                        changed = True

            merged.append(current)

        #  id
        for idx, m in enumerate(merged):
            m["id"] = idx
        return merged

    def _detect_image_regions(self, img, text_mask, w: int, h: int) -> List[Dict]:
        """
        

        
        1.     
        2. 
        3. 
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5
        )

        # 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        # 
        text_expanded = cv2.dilate(text_mask, kernel, iterations=1)
        non_text = cv2.bitwise_and(dilated, cv2.bitwise_not(text_expanded))

        # 
        contours, _ = cv2.findContours(non_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_area = w * h
        min_area = total_area * self.min_image_area_ratio

        image_blocks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            px1, py1, bw, bh = cv2.boundingRect(cnt)
            px2, py2 = px1 + bw, py1 + bh

            #  [0, 1000]
            bbox = [
                px1 / w * 1000,
                py1 / h * 1000,
                px2 / w * 1000,
                py2 / h * 1000,
            ]

            image_blocks.append({
                "type": "image",
                "bbox": bbox,
                "content": "",
                "page_idx": 0,
            })

        return image_blocks
