from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
ATTENDANCE_FILE = BASE_DIR / "attendance.csv"

app = FastAPI(title="Smart Attendance Backend", version="1.0.0")


class MarkAttendanceRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    timestamp: Optional[datetime] = None


def _ensure_storage() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ATTENDANCE_FILE.touch(exist_ok=True)


def _safe_name(raw_name: str) -> str:
    # Keep names filesystem-safe and consistent with existing uppercase matching.
    cleaned = "".join(ch for ch in raw_name.strip() if ch.isalnum() or ch in ("_", "-", " "))
    cleaned = cleaned.replace(" ", "_")
    if not cleaned:
        raise HTTPException(status_code=400, detail="Invalid name provided")
    return cleaned.upper()


def _read_attendance_rows() -> list[list[str]]:
    _ensure_storage()
    rows: list[list[str]] = []
    with ATTENDANCE_FILE.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 3:
                rows.append(row[:3])
    return rows


@app.on_event("startup")
def startup_event() -> None:
    _ensure_storage()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/students")
def list_students() -> dict[str, list[str]]:
    _ensure_storage()
    students = sorted({image.stem.upper() for image in IMAGES_DIR.iterdir() if image.is_file()})
    return {"students": students}


@app.post("/students")
async def add_student(name: str = Query(..., min_length=1), image: UploadFile = File(...)) -> dict[str, str]:
    _ensure_storage()
    student_name = _safe_name(name)

    if not image.filename:
        raise HTTPException(status_code=400, detail="Image filename is required")

    suffix = Path(image.filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Only .jpg, .jpeg, and .png files are allowed")

    destination = IMAGES_DIR / f"{student_name}{suffix}"
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    destination.write_bytes(content)
    return {"message": "Student image saved", "name": student_name, "file": destination.name}


@app.get("/attendance")
def get_attendance(date: Optional[str] = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")) -> dict[str, list[dict[str, str]]]:
    rows = _read_attendance_rows()
    data = [
        {"name": row[0], "date": row[1], "time": row[2]}
        for row in rows
        if date is None or row[1] == date
    ]
    return {"attendance": data}


@app.post("/attendance/mark")
def mark_attendance(payload: MarkAttendanceRequest) -> dict[str, str]:
    _ensure_storage()
    name = _safe_name(payload.name)
    now = payload.timestamp or datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    rows = _read_attendance_rows()
    already_marked = any(row[0] == name and row[1] == date for row in rows)
    if already_marked:
        return {"message": "Already marked", "name": name, "date": date}

    with ATTENDANCE_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([name, date, time])

    return {"message": "Attendance marked", "name": name, "date": date, "time": time}


@app.get("/attendance/summary")
def attendance_summary(date: Optional[str] = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")) -> dict[str, int]:
    rows = _read_attendance_rows()
    if date is not None:
        rows = [row for row in rows if row[1] == date]

    unique_students = {row[0] for row in rows}
    return {
        "total_entries": len(rows),
        "unique_students": len(unique_students),
    }
