import csv
import re
from pathlib import Path


# Rutas configuradas manualmente
INPUT_DIR = Path(r"C:\Users\Human\Desktop\IZB\Proyectos\HIQ\Estudios\HIQ_preprocess\data\sourcedata\eeg")
OUTPUT_FILE = Path(r"C:\Users\Human\Desktop\IZB\Proyectos\HIQ\Estudios\HIQ_preprocess\data\recordings.tsv")

PATTERN = re.compile(
    r"^HIQ_([0-9]{3})_([0-9]+)_(1EC)_.*\.vhdr$",
    flags=re.IGNORECASE,
)


def create_recordings_tsv() -> None:
    if OUTPUT_FILE.exists():
        if not OUTPUT_FILE.is_file():
            raise RuntimeError(
                f"La ruta de salida existe, pero no es un archivo: {OUTPUT_FILE}"
            )

        OUTPUT_FILE.unlink()
        print(f"Archivo anterior eliminado: {OUTPUT_FILE}")

    rows = []

    for file in sorted(INPUT_DIR.iterdir()):
        if not file.is_file():
            continue

        match = PATTERN.fullmatch(file.name)
        if match is None:
            continue

        subject, session, task = match.groups()

        rows.append(
            {
                "file": file.name,
                "subject": subject,
                "session": session,
                "task": task,
                "run": "0",
            }
        )

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as tsv_file:
        writer = csv.DictWriter(
            tsv_file,
            fieldnames=["file", "subject", "session", "task", "run"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Archivo creado: {OUTPUT_FILE}")
    print(f"Grabaciones incluidas: {len(rows)}")


if __name__ == "__main__":
    create_recordings_tsv()