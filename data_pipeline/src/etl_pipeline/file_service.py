import json
from hashlib import sha256
from pathlib import Path

from data_pipeline.src.constants import PARSE_PARAMS

BASE_DIR = Path(__file__).resolve().parents[3]

raw_path = BASE_DIR / "data_pipeline" / "data" / "raw"
processed_path = BASE_DIR / "data_pipeline" / "data" / "processed"
eval_path = BASE_DIR / "data_pipeline" / "data" / "eval" / "questions.json"


def get_raw_files():

    files = list(raw_path.glob("*.pdf"))
    return files


def get_processed_files():

    files = list(processed_path.glob("*.json"))
    return files


def get_processed_file(name: str, file_hash: str):
    file_path = processed_path / f"{name}.{file_hash}.json"
    return file_path


def get_file_hash(pdf_path: Path, limit: int = 12):

    file_bytes = pdf_path.read_bytes()

    params_bytes = json.dumps(PARSE_PARAMS, sort_keys=True).encode()

    full_bytes = file_bytes + params_bytes
    hash_str = sha256(full_bytes).hexdigest()

    return hash_str[:limit]
