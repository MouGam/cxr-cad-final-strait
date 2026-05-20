"""
DICOM 샘플 파싱 및 PHI 필드 확인 데모.

NIH ChestX-ray14 본 데이터는 PNG로 배포되지만, 과제 요구사항의 DICOM 파싱
실습을 재현하기 위해 pydicom 내장 샘플을 읽고 주요 메타데이터와 pixel array를
확인한다. 실제 환자 DICOM을 사용할 경우 PatientName, PatientID 등 PHI 필드는
제거한 뒤 비공개로 관리해야 한다.
"""

from __future__ import annotations

from pathlib import Path

import pydicom
from pydicom import examples


PHI_FIELDS = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "InstitutionName",
    "AccessionNumber",
]


def main() -> None:
    ds = examples.ct
    output_dir = Path("report_assets")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "03_dicom_parsing_example.txt"

    lines = [
        "pydicom DICOM metadata parsing result",
        "",
        f"Modality: {getattr(ds, 'Modality', '')}",
        f"Rows x Columns: {ds.Rows} x {ds.Columns}",
        f"Bits Allocated: {getattr(ds, 'BitsAllocated', '')}",
        f"Pixel Array Shape: {ds.pixel_array.shape}",
        "",
        "[PHI fields]",
    ]

    for field in PHI_FIELDS:
        lines.append(f"{field}: {getattr(ds, field, '<missing>')}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
