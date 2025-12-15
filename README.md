# 内容

PyMuPDF4LLM は、見開きPDFの分割から高解像度画像生成、RapidOCR・Docling・vLLMを組み合わせたOCR自動化までを一体化した実験用パイプラインです。PyMuPDFとOpenCVでスパイン推定やレイアウト補正を行い、books/<book_id> 配下に spreads/pages/outputs を生成して差分再処理を容易にします。LLMクライアント群はMinistral系モデルをローカルvLLMサーバーへ送信し、推論過程と回答の双方を収集可能です。各スクリプトを組み合わせることで、大量の技術書・資料PDFを高速にMarkdownへ落とし込み、追加解析や要約にすぐ回せる運用を目指しています。

## split_two-page_spread.py

このスクリプトは、見開きPDFを自動分割し、RapidOCRを用いてDoclingでOCR（文字認識）を行い、Markdown形式のテキストを出力する処理を自動化します。`fitz`でページ比率を判定し、`pikepdf`で左右ページに分割後、新しいPDFを生成。Docling経由でOCR処理を実行し、結果をテキスト（および必要に応じてメタデータJSON）として保存します。高精度RapidOCR（ONNXモデル）をGPUで実行する設定も組み込まれています。

---

### 実行方法の例

以下のように、入力PDFを指定して実行します。結果は自動的に `output/` フォルダに保存されます。

```bash
# 単純な実行（1.pdf を処理）
python split_two-page_spread.py ./1.pdf
```

```bash
# 分割比率を調整したい場合（中央より少し右寄りで分割）
python split_two-page_spread.py ./book.pdf --split-ratio 0.53
```

```bash
# 和書など右開きの本の場合（右ページが若い）
python split_two-page_spread.py ./novel.pdf --right-open
```

```bash
# OCRエンジンを変更（例：Tesseractを使用）
python split_two-page_spread.py ./1.pdf --backend tesseract
```

---

### コマンドライン引数・設定項目（表形式）

| 項目                  | 内容                                              | 既定値／例                      |
| ------------------- | ----------------------------------------------- | -------------------------- |
| `input_pdf`         | 入力PDFファイル名                                      | `"1.pdf"`                  |
| `work_dir`          | 出力ディレクトリ                                        | `"output"`                 |
| `left_is_older`     | 左ページが若いか（右開き書籍はFalse）                           | `True`                     |
| `split_ratio`       | 分割位置の比率（0.0〜1.0）                                | `0.535`                    |
| `backend`           | OCRエンジン選択（`rapidocr` / `easyocr` / `tesseract`） | `"rapidocr"`               |
| `SAVE_DOC_METADATA` | Doclingの構造データをJSON保存するか                         | `False`                    |
| `meta_json`         | メタデータ保存先（有効時）                                   | `output/1_split_meta.json` |

出力結果は `output/1_split.pdf`（見開き分割後PDF）と `output/1_ocr.txt`（OCR結果テキスト）として保存され、必要に応じて `output/1_split_meta.json` にDocling構造情報も出力されます。

___

## pre-process-split.py

高解像度の見開きPDFを、指定ディレクトリ配下に spreads/pages/outputs として展開し直す前処理スクリプトです。PyMuPDFでベクタ保持したまま左右ページを生成し、OpenCV（`--use-cv`）を有効にするとスパイン候補を複数手法で推定して信頼度順に採用します。比率解析で片面ページも判別でき、右開き順やオーバーラップ量、デバッグオーバーレイなども調整可能です。

### 実装例

```bash
python pre-process-split.py ./1.pdf --book-dir books/sample --use-cv
```

```bash
python pre-process-split.py ./mags.pdf --book-dir books/mag001 --dpi 600 --overlap 12 --export-pdf
```

### コマンドライン引数・設定項目（表形式）

| 項目               | 内容                                                       | 既定値／例                |
| ------------------ | ---------------------------------------------------------- | ------------------------- |
| `pdf`              | 入力見開きPDFパス                                          | `./1.pdf`                 |
| `--book-dir`       | spreads/pages/outputs を作成する書籍ルート（必須）         | `books/<book_id>`         |
| `--dpi`            | PNGレンダリング解像度                                      | `400`                     |
| `--rtl`            | 右開き書籍としてページ順を逆転                             | `False`                   |
| `--overlap`        | スパイン跨ぎオーバーラップ幅（72dpi基準pixel）            | `0`                       |
| `--use-cv`         | OpenCVベースの自動スパイン推定を有効化                     | `False`                   |
| `--export-pdf`     | 分割結果を `pages_split.pdf` に統合出力                    | `False`                   |
| `--min-confidence` | CV推定採用の最小信頼度                                     | `0.55`                    |
| `--debug-overlay`  | 検出矩形を `outputs/debug_overlays` へ保存                  | `False`                   |

---

## extract_spreads.py

単一の見開き画像から左右ページ領域を抽出し、`<stem>_Left.png` / `_Right.png` に書き出す軽量ユーティリティです。CLAHE正規化と自適応二値化を行い、投影・ハフ検出・輪郭クラスターの三手法でスパイン候補を出し、中央値的に合意形成するためノイズや罫線にも比較的強い設計になっています。OpenCVのみで完結するため前処理結果の素早い検証にも向きます。

### 実装例

```bash
python extract_spreads.py ./spreads/0003.png
```

```bash
python extract_spreads.py scans/magazine_spread.jpg
```

### コマンドライン引数・設定項目（表形式）

| 項目          | 内容                               | 既定値／例             |
| ------------- | ---------------------------------- | ---------------------- |
| `image_path`  | 見開き画像（PNG/JPG/TIFFなど）     | `./spreads/0001.png`   |

---

## extract_spreads_hough_variants.py

ハフ変換ベースの複数ストラテジー（通常版・長辺優先・強ブラー抑制など）を切り替えつつ、最良の左右矩形を投票で決める高度版抽出器です。`--strategies` で探索手法を組み合わせ、各候補の信頼度を比較しながら自動保存します。ディレクトリ一括処理やマージン保持、デバッグ可視化が行えるため、OCR前の検品バッチに組み込みやすい構成です。

### 実装例

```bash
python extract_spreads_hough_variants.py ./spreads/0004.png --debug
```

```bash
python extract_spreads_hough_variants.py ./raw_scans --out-dir ./out_spreads --strategies hough,hough_blur
```

### コマンドライン引数・設定項目（表形式）

| 項目             | 内容                                                    | 既定値／例                               |
| ---------------- | ------------------------------------------------------- | ---------------------------------------- |
| `input`          | 画像または画像ディレクトリ                             | `./spreads/0004.png`                     |
| `--out-dir`      | 左右ページPNGを書き出す先                              | `./out_spreads`                          |
| `--keep-margin`  | 仕上がり画像に残す外周マージンpixel                    | `8`                                      |
| `--strategies`   | 利用するストラテジー列挙（カンマ区切り）                | `hough,hough_long,hough_blur`            |
| `--debug`        | スパイン線と矩形を重畳したデバッグ画像を合わせて保存    | `False`                                  |

---

## mistral.py

vLLM(OpenAI互換API) に画像+テキストプロンプトを送り、Ministral推論結果をストリーミング受信できる最小クライアントです。ローカルで提供中のモデル一覧からデフォルトを自動取得し、Reasoning Effortや温度、トークン長を細かく制御できます。LLMの思考過程と回答を別セクションで整形出力するため、OCRプロンプト検証やプロンプトチューニング時の比較が容易です。

### 実装例

```bash
python mistral.py ./output/pages/0001/0001.png
```

```bash
python mistral.py ./output/pages/0002/0002.png --temperature 0.2 --no_stream --max_tokens 4096
```

### コマンドライン引数・設定項目（表形式）

| 項目             | 内容                                                           | 既定値／例                               |
| ---------------- | -------------------------------------------------------------- | ---------------------------------------- |
| `image_path`     | vLLMへ送るページ画像                                          | `./output/pages/0001/0001.png`           |
| `--model`        | 使用するモデルID（空なら /v1/models の先頭を採用）            | `""`                                    |
| `--max_tokens`   | 生成上限トークン                                               | `8192`                                   |
| `--temperature`  | サンプリング温度                                               | `0.0`                                    |
| `--no_stream`    | ストリーミングを無効化して一括応答にする                      | `False`                                  |
| `--effort`       | Reasoning Effort（`low`/`medium`/`high`）                      | `medium`                                 |

---

## mistral_ocr.py

`pre-process-split.py` と連携しながら `1.pdf` を分割・PNG化し、Ministral推論でページ単位のMarkdown OCRを保存する自動バッチです。Spread画像を優先的に再利用し、未生成時のみPyMuPDFで再レンダリングして差分更新します。Reasoning出力を含むテキストを `output/pages/<page>/` に逐次書き込むため、長編PDFでも途中結果を確認しながら処理を継続できます。

### 実装例

```bash
python mistral_ocr.py
```

```bash
API_BASE=http://127.0.0.1:8002/v1 MODEL_NAME=my-model python mistral_ocr.py
```

### コマンドライン引数・設定項目（表形式）

| 項目     | 内容                                             | 既定値／例                         |
| -------- | ------------------------------------------------ | ---------------------------------- |
| *(なし)* | すべてスクリプト内部の定数で制御（`API_BASE` など環境変数で上書き可) | `python mistral_ocr.py` 実行のみ |
