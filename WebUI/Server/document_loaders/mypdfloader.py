from typing import List
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from WebUI.Server.document_loaders.ocr import get_ocr
import tqdm

class RapidOCRPDFLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def pdf2text(filepath):
            import fitz
            import numpy as np
            ocr = get_ocr()
            doc = fitz.open(filepath)
            resp = ""
            print("RapidOCRPDFLoader: ", filepath)
            b_unit = tqdm.tqdm(total=doc.page_count, desc="RapidOCRPDFLoader context page index: 0")
            for i, page in enumerate(doc):

                b_unit.set_description("RapidOCRPDFLoader context page index: {}".format(i))
                b_unit.refresh()
                text = page.get_text("")
                resp += text + "\n"

                img_list = page.get_images()
                for img in img_list:
                    try:
                        pix = fitz.Pixmap(doc, img[0])
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                        result, _ = ocr(img_array)
                        if result:
                            ocr_result = [line[1] for line in result]
                            resp += "\n".join(ocr_result)
                    except Exception as e:
                        print(f"image error: {e}")
                b_unit.update(1)
            return resp

        text = pdf2text(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(text=text, **self.unstructured_kwargs)


if __name__ == "__main__":
    loader = RapidOCRPDFLoader(file_path="../tests/samples/ocr_test.pdf")
    docs = loader.load()
    print(docs)
