FROM paddlepaddle/paddle:3.1.0-gpu-cuda12.6-cudnn9.5

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /work

# PDFレンダリング・画像処理向け（PyMuPDFはpipで入れる）
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    fonts-ipafont \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U pip setuptools wheel
RUN python -m pip install "paddleocr[all]" pymupdf pillow numpy

# 実行スクリプトをコンテナへ
COPY batch_ocr.py /work/batch_ocr.py

CMD ["python", "/work/batch_ocr.py", "--in_dir", "/work/input", "--out_dir", "/work/output", "--lang", "japan"]
