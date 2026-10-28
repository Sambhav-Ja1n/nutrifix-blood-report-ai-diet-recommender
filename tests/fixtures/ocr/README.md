# OCR smoke corpus

The OCR tests generate synthetic report-like content in memory; no real reports or personal data are stored. Expected fields are asserted directly in tests/test_report_extraction.py.

Synthetic cases cover selectable-text PDFs, image-only scanned PDFs, PNG and JPEG inputs, a mixed two-page PDF, decimal values, units and supplied ranges, RBC/RDW name distinction, an ambiguous unmapped Neutrophils label, conflicting repeated values, a decimal-comma value, missing units, blank images, corrupt input, empty input, and unsupported media types.

These are deterministic smoke tests on synthetic examples. They are not a representative, manually annotated clinical corpus and do not support field-level precision, recall, or F1 claims.
