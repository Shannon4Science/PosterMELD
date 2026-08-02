"""
extract_mineru.py   MinerU  API  poster 

MinerU API : https://mineru.net/apiManage/docs
:
  1.  URL
  2. 
  3. 
  4. 
"""

import os
import json
import time
import zipfile
import tempfile
import subprocess
import requests
from typing import List, Dict, Optional


class MinerUExtractor:
    """MinerU  API """

    def __init__(self, token: str, base_url: str = "https://mineru.net/api/v4",
                 model_version: str = "vlm", is_ocr: bool = True,
                 enable_table: bool = True, enable_formula: bool = True,
                 language: str = "en", poll_interval: int = 3,
                 max_poll_time: int = 300):
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.model_version = model_version
        self.is_ocr = is_ocr
        self.enable_table = enable_table
        self.enable_formula = enable_formula
        self.language = language
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        #  Session 
        self.session = requests.Session()
        self.session.trust_env = False  #  Windows 

    def extract(self, image_path: str) -> List[Dict]:
        """
         poster 

        Args:
            image_path: poster 

        Returns:
            list of dict,  dict :
            - id: int, 
            - type: str, 'text' | 'image' | 'table' | 'equation' 
            - bbox: [x1, y1, x2, y2],  [0, 1000]
            - content: str, text 
            - page_idx: int, 
        """
        # Step 1:  URL
        file_name = os.path.basename(image_path)
        batch_id, upload_url = self._get_upload_url(file_name)

        # Step 2: 
        self._upload_file(upload_url, image_path)

        # Step 3: 
        result_url = self._poll_batch_result(batch_id)

        # Step 4: 
        blocks = self._download_and_parse(result_url)

        return blocks

    def _get_upload_url(self, file_name: str):
        """ URL"""
        resp = self.session.post(
            f"{self.base_url}/file-urls/batch",
            headers=self.headers,
            json={
                "files": [{"name": file_name}],
                "model_version": self.model_version,
                "is_ocr": self.is_ocr,
                "enable_table": self.enable_table,
                "enable_formula": self.enable_formula,
                "language": self.language,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"MinerU API : {data.get('msg', data)}")

        batch_id = data["data"]["batch_id"]
        upload_url = data["data"]["file_urls"][0]
        print(f"[MinerU] batch_id={batch_id}, ...")
        return batch_id, upload_url

    def _upload_file(self, upload_url: str, file_path: str):
        """ URL curl -T  Python SSL """
        import shutil
        file_path = os.path.abspath(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)

        # 
        tmp_dir = tempfile.mkdtemp()
        tmp_file = os.path.join(tmp_dir, f"upload{ext}")
        shutil.copy2(file_path, tmp_file)

        try:
            cmd = [
                "curl", "-s", "-w", "\nHTTP_CODE:%{http_code}",
                "-T", tmp_file,
                "--connect-timeout", "30",
                "--max-time", "120",
                upload_url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=130)
            #  HTTP 
            status_code = ""
            for line in result.stdout.strip().split("\n"):
                if line.startswith("HTTP_CODE:"):
                    status_code = line.split(":")[1]
            if status_code.startswith("2"):
                print(f"[MinerU]  ({file_size / 1024:.1f} KB)")
            else:
                raise RuntimeError(
                    f", HTTP {status_code}\n{result.stdout[:500]}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("curl ")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _poll_batch_result(self, batch_id: str) -> str:
        """"""
        url = f"{self.base_url}/extract-results/batch/{batch_id}"
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.max_poll_time:
                raise TimeoutError(
                    f"MinerU  ({self.max_poll_time}s)")

            resp = self.session.get(url, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") != 0:
                raise RuntimeError(f"MinerU API : {data.get('msg', data)}")

            extract_result = data["data"]["extract_result"]
            if not extract_result:
                print(f"[MinerU] ... ({elapsed:.0f}s)")
                time.sleep(self.poll_interval)
                continue

            state = extract_result[0].get("state")
            if state == "done":
                full_zip_url = extract_result[0]["full_zip_url"]
                print(f"[MinerU] ! ({elapsed:.0f}s)")
                return full_zip_url
            elif state == "failed":
                raise RuntimeError(
                    f"MinerU : {extract_result[0].get('err_msg', '')}")
            else:
                print(f"[MinerU] : {state}, ... ({elapsed:.0f}s)")
                time.sleep(self.poll_interval)

    def _download_and_parse(self, zip_url: str) -> List[Dict]:
        """ ZIP  block """
        blocks = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "result.zip")

            #  curl /SSL 
            cmd = ["curl", "-s", "-o", zip_path, "--connect-timeout", "30",
                   "--max-time", "120", zip_url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=130)
            if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
                raise RuntimeError(f" ZIP : {result.stderr[:300]}")

            print(f"[MinerU]  ({os.path.getsize(zip_path) / 1024:.1f} KB)")

            # 
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)

            #  content_list.json [0, 1000]
            content_list_path = self._find_file(tmp_dir, "content_list.json")
            if content_list_path:
                blocks = self._parse_content_list(content_list_path)
            else:
                #  middle.json
                middle_path = self._find_file(tmp_dir, "middle.json")
                if middle_path:
                    blocks = self._parse_middle_json(middle_path)
                else:
                    print("[MinerU] :  content_list.json  middle.json")

        print(f"[MinerU]  {len(blocks)} ")
        return blocks

    def _find_file(self, directory: str, filename: str) -> Optional[str]:
        """MinerU  UUID """
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.endswith(filename) or filename in f:
                    return os.path.join(root, f)
        return None

    def _parse_content_list(self, json_path: str) -> List[Dict]:
        """
         content_list.json

        content_list.json :
        [
            {
                "type": "text",
                "bbox": [x1, y1, x2, y2],  #  [0, 1000]
                "text": "...",
                "page_idx": 0,
                "text_level": 0  # 0=, 1/2/...=
            },
            ...
        ]
        """
        with open(json_path, "r", encoding="utf-8") as f:
            content_list = json.load(f)

        blocks = []
        for i, item in enumerate(content_list):
            item_type = item.get("type", "text")
            bbox = item.get("bbox", [0, 0, 0, 0])

            #  bbox
            if bbox == [0, 0, 0, 0] or len(bbox) != 4:
                continue

            block = {
                "id": i,
                "type": self._normalize_type(item_type),
                "bbox": bbox,  # [0, 1000] 
                "content": item.get("text", ""),
                "page_idx": item.get("page_idx", 0),
            }

            #  text_level
            text_level = item.get("text_level", 0)
            if text_level > 0:
                block["is_title"] = True
                block["title_level"] = text_level

            blocks.append(block)

        return blocks

    def _parse_middle_json(self, json_path: str) -> List[Dict]:
        """ middle.json"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        blocks = []
        block_id = 0

        for page in data.get("pdf_info", []):
            page_idx = page.get("page_idx", 0)
            page_w = page.get("page_size", [1, 1])[0]
            page_h = page.get("page_size", [1, 1])[1]

            for pre_block in page.get("preproc_blocks", []):
                bbox = pre_block.get("bbox", [0, 0, 0, 0])
                block_type = pre_block.get("type", "text")

                #  [0, 1000]
                norm_bbox = [
                    bbox[0] / page_w * 1000,
                    bbox[1] / page_h * 1000,
                    bbox[2] / page_w * 1000,
                    bbox[3] / page_h * 1000,
                ]

                # 
                text_parts = []
                for line in pre_block.get("lines", []):
                    for span in line.get("spans", []):
                        text_parts.append(span.get("content", ""))

                blocks.append({
                    "id": block_id,
                    "type": self._normalize_type(block_type),
                    "bbox": norm_bbox,
                    "content": " ".join(text_parts),
                    "page_idx": page_idx,
                })
                block_id += 1

        return blocks

    @staticmethod
    def _normalize_type(raw_type: str) -> str:
        """"""
        type_map = {
            "text": "text",
            "title": "text",
            "header": "text",
            "footer": "text",
            "list": "text",
            "code": "text",
            "aside_text": "text",
            "page_footnote": "text",
            "image": "image",
            "image_body": "image",
            "chart": "image",
            "table": "table",
            "table_body": "table",
            "equation": "equation",
            "interline_equation": "equation",
        }
        return type_map.get(raw_type, "text")
