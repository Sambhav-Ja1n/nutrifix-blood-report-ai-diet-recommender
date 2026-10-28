"""One-shot subprocess entry point for bounded report extraction."""

from __future__ import annotations

import json
import sys

from src.nutrifix.report_extraction import (
    MAX_UPLOAD_BYTES,
    OCRUnavailableError,
    ReportExtractionError,
    extract_report,
)


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    media_type = sys.argv[1]
    content = sys.stdin.buffer.read(MAX_UPLOAD_BYTES + 1)
    try:
        result = extract_report(content, media_type)
        payload = {"status_code": 200, "result": result}
    except OCRUnavailableError:
        payload = {
            "status_code": 503,
            "detail": "Local OCR is unavailable. Try a selectable-text PDF or contact the service owner.",
        }
    except ReportExtractionError as exc:
        payload = {"status_code": 400, "detail": str(exc)}
    except Exception:
        payload = {"status_code": 400, "detail": "Could not process this report."}

    sys.stdout.write(json.dumps(payload, allow_nan=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
