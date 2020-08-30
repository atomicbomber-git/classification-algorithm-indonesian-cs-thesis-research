import os
import csv
import re
from tika import parser
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

INPUT_DIR = "/home/atomicbomber/Downloads/DATA_TA_JEFRI/justin"
OUTPUT_FILE = "./output.csv"

ALLOWED_LABELS = [
    "SI",
    "komputasi",
    "jaringan",
]


def clean_text(input_text: str) -> str:
    # Filtering, menghapus semua karakter non teks
    input_text = re.sub(r'\W', ' ', input_text)

    # Menghapus semua karakter tunggal pada bagian tengah teks
    input_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', input_text)

    # Menghapus semua karakter tunggal pada awal teks
    input_text = re.sub(r'\^[a-zA-Z]\s+', ' ', input_text)

    # Mengganti spasi berurutan dengan ' '
    input_text = re.sub(r'\s+', ' ', input_text, flags=re.I)

    # Case folding
    input_text = input_text.lower()

    # Stemming
    input_text = stemmer.stem(input_text)
    return input_text


with open(OUTPUT_FILE, mode="w") as output_file:
    output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    output_writer.writerow(["data", "target"])

    for root_directory, sub_dirs, file_names in os.walk(INPUT_DIR):
        for file_name in file_names:

            supposed_label = file_name.split("-")[0]
            label = None

            for allowed_label in ALLOWED_LABELS:
                if allowed_label in supposed_label:
                    label = allowed_label
                    break

            if not label:
                raise ValueError

            text = parser.from_file(
                os.path.join(
                    root_directory,
                    file_name
                )
            )["content"]

            text = clean_text(text)

            output_writer.writerow([text, label])
