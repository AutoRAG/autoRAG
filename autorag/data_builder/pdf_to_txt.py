import os
from pathlib import Path
from tqdm import tqdm
import argparse
from pypdf import PdfReader


def parse_single_pdf(pdf_path):
    """
    Convert a single pdf to text. Return text.
    """
    # Load your PDF
    reader = PdfReader(pdf_path)
    number_of_pages = len(reader.pages)
    return " ".join([page.extract_text() for page in reader.pages])


def pdf_to_txt(pdf_dir, txt_dir):
    """
    For each pdf in the pdf_dir, convert to text and save in txt_dir.
    Note that it will simply overwrite for now.

    Assumes that every file in pdf_dir ends in .pdf needs to be converted to .txt.
    """
    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)

    for root, dirs, files in tqdm(os.walk(pdf_dir)):
        txt_root = os.path.join(txt_dir, os.path.relpath(root, pdf_dir))

        txt_root_exists = os.path.exists(txt_root)

        for file in files:
            if file.lower().endswith(".pdf"):

                if not txt_root_exists:
                    os.makedirs(txt_root)

                pdf_path = os.path.join(root, file)
                filename = Path(pdf_path).stem
                txt_path = os.path.join(txt_root, filename + ".txt")

                if not os.path.exists(txt_path):
                    single_pdf_to_text(pdf_path, txt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pdf_dir", required=True, help="Path to directory containing pdfs.", type=str
    )
    parser.add_argument(
        "--txt_dir", required=True, help="Path to directory storing txts.", type=str
    )

    args = parser.parse_args()

    pdf_dir = args.pdf_dir
    txt_dir = args.txt_dir

    pdf_to_txt(pdf_dir, txt_dir)
