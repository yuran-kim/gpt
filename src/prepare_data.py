import os
import requests

DATA_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
RAW_PATH = os.path.join("data", "raw.txt")
INPUT_PATH = os.path.join("data", "input.txt")
START_MARKER = "It is a truth universally acknowledged"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***"


def download_text(url: str) -> str:
    """주어진 URL에서 텍스트를 다운로드하여 문자열로 반환합니다."""
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"데이터 다운로드 실패: HTTP {response.status_code}")
    response.encoding = "utf-8"
    return response.text


def save_text(path: str, text: str) -> None:
    """UTF-8로 텍스트를 파일에 저장합니다."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def extract_body(raw_text: str) -> str:
    """원본 텍스트에서 시작과 종료 마커 사이의 본문을 추출합니다."""
    start_idx = raw_text.find(START_MARKER)
    if start_idx == -1:
        raise ValueError(f"시작 마커를 찾을 수 없습니다: {START_MARKER}")
    end_idx = raw_text.find(END_MARKER)
    if end_idx == -1:
        raise ValueError(f"종료 마커를 찾을 수 없습니다: {END_MARKER}")
    return raw_text[start_idx:end_idx].rstrip() + "\n"


def main() -> None:
    """데이터를 다운로드하고 정제된 입력 파일을 생성합니다."""
    raw_text = download_text(DATA_URL)
    save_text(RAW_PATH, raw_text)

    body_text = extract_body(raw_text)
    save_text(INPUT_PATH, body_text)

    unique_chars = sorted(set(body_text))
    print(f"원본 문자 수: {len(raw_text)}")
    print(f"정제 후 문자 수: {len(body_text)}")
    print(f"고유 문자 수: {len(unique_chars)}")


if __name__ == "__main__":
    main()
