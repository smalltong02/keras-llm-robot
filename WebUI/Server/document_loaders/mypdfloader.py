import cv2
from PIL import Image
import numpy as np
from typing import List
from WebUI.configs.kbconfig import PDF_OCR_THRESHOLD
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from WebUI.Server.document_loaders.ocr import get_ocr
import tqdm

class RapidOCRPDFLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def rotate_img(img, angle):
            '''
            img   --image
            angle --rotation angle
            return--rotated img
            '''
            
            h, w = img.shape[:2]
            rotate_center = (w/2, h/2)
            M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
            new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
            new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2

            rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
            return rotated_img
        
        def pdf2text(filepath):
            import fitz
            import numpy as np
            ocr = get_ocr()
            doc = fitz.open(filepath)
            resp = ""

            b_unit = tqdm.tqdm(total=doc.page_count, desc="RapidOCRPDFLoader context page index: 0")
            for i, page in enumerate(doc):
                b_unit.set_description("RapidOCRPDFLoader context page index: {}".format(i))
                b_unit.refresh()
                text = page.get_text("")
                resp += text + "\n"

                img_list = page.get_image_info(xrefs=True)
                for img in img_list:
                    if xref := img.get("xref"):
                        bbox = img["bbox"]
                        if ((bbox[2] - bbox[0]) / (page.rect.width) < PDF_OCR_THRESHOLD[0]
                            or (bbox[3] - bbox[1]) / (page.rect.height) < PDF_OCR_THRESHOLD[1]):
                            continue
                        pix = fitz.Pixmap(doc, xref)
                        samples = pix.samples
                        if int(page.rotation)!=0:
                            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                            tmp_img = Image.fromarray(img_array);
                            ori_img = cv2.cvtColor(np.array(tmp_img),cv2.COLOR_RGB2BGR)
                            rot_img = rotate_img(img=ori_img, angle=360-page.rotation)
                            img_array = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
                        else:
                            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)

                        result, _ = ocr(img_array)
                        if result:
                            ocr_result = [line[1] for line in result]
                            resp += "\n".join(ocr_result)
                b_unit.update(1)
            return resp

        text = pdf2text(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(text=text, **self.unstructured_kwargs)


if __name__ == "__main__":
    loader = RapidOCRPDFLoader(file_path="../tests/samples/ocr_test.pdf")
    docs = loader.load()
    print(docs)
