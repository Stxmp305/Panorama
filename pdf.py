"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AI Invoice OCR & Extractor — CLI & Web API                           ║
║         สกัดข้อมูลจาก PDF Invoice เป็น JSON — รองรับทั้ง Text และ Image         ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEPENDENCIES:
    pip install requests pdf2image pillow rich pypdf fastapi uvicorn
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional, Any

# ── Windows UTF-8 Console Reconfiguration ─────────────────────────────────────
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ── Dependency Imports ────────────────────────────────────────────────────────
try:
    import requests
except ImportError:
    sys.exit("❌ 'requests' not found. Run: pip install requests")

try:
    from pdf2image import convert_from_path
except ImportError:
    sys.exit("❌ 'pdf2image' not found. Run: pip install pdf2image pillow")

try:
    from PIL import Image
except ImportError:
    sys.exit("❌ 'Pillow' not found. Run: pip install pillow")

try:
    import pypdf
except ImportError:
    sys.exit("❌ 'pypdf' not found. Run: pip install pypdf")

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    import shutil
    HAS_WEB_API = True
except ImportError:
    HAS_WEB_API = False

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b"  # ปลอดภัยกว่า gemma4:e2b สำหรับ VRAM 4GB
DEFAULT_DPI = 180
MAX_IMAGE_PX = 1280
JPEG_QUALITY = 88
REQUEST_TIMEOUT = 300

console = Console() if HAS_RICH else None

# ── Pretty Print Helpers ──────────────────────────────────────────────────────
def info(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[cyan]ℹ[/cyan]  {msg}")
    else:
        print(f"ℹ  {msg}")

def success(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[green]✅[/green]  {msg}")
    else:
        print(f"✅  {msg}")

def warn(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[yellow]⚠️[/yellow]   {msg}")
    else:
        print(f"⚠️   {msg}")

def error_exit(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[bold red]❌  {msg}[/bold red]")
    else:
        print(f"❌  {msg}")
    sys.exit(1)

# ── Tkinter GUI File Picker Fallback ──────────────────────────────────────────
def select_file_gui() -> Optional[Path]:
    """เปิดหน้าต่าง GUI ให้ผู้ใช้เลือกไฟล์ PDF"""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    root = tk.Tk()
    root.withdraw()  # ซ่อนหน้าต่างหลัก
    root.attributes('-topmost', True)  # ดึงมาด้านบนสุด

    file_path = filedialog.askopenfilename(
        title="เลือกไฟล์ PDF Invoice",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
    )
    root.destroy()
    return Path(file_path) if file_path else None

# ── JSON Parser Helper ────────────────────────────────────────────────────────
def clean_and_parse_json(raw_text: str) -> dict[str, Any]:
    """สกัดและแปลงข้อความที่ได้จาก LLM ให้เป็น JSON/Dictionary ที่ถูกต้อง"""
    cleaned = raw_text.strip()
    
    # ลบ markdown tick blocks ออกถ้ามี
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()

    # ค้นหาบล็อก { ... } แรกและสุดท้าย
    match = re.search(r'(\{.*\})', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # ลองลบ trailing commas เพื่อกู้คืนโครงสร้าง JSON ในกรณีโมเดลตอบผิดไวยากรณ์เล็กน้อย
        try:
            cleaned_adjusted = re.sub(r',\s*([}\]])', r'\1', cleaned)
            return json.loads(cleaned_adjusted)
        except Exception:
            # คืนค่า default schema หากการ parse ล้มเหลวทั้งหมด
            return {
                "tax_provider": None,
                "tax_merchant": None,
                "date": None,
                "document_id": None,
                "merchant_address": None,
                "merchant_branch": None,
                "total_discount_amount": None,
                "total_value_added_tax": None,
                "raw_response_error": raw_text
            }

# ── Ollama Health Checker ─────────────────────────────────────────────────────
def check_ollama(host: str, model: str) -> None:
    """ตรวจสอบความพร้อมของ Ollama และโมเดล"""
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        error_exit("ไม่พบ Ollama — กรุณารัน 'ollama serve' ในอีกหน้าต่างก่อน")
    except Exception as exc:
        error_exit(f"Ollama health check failed: {exc}")

    models_available = [m["name"] for m in r.json().get("models", [])]
    model_base = model.split(":")[0]
    found = any(m == model or m.startswith(model_base + ":") for m in models_available)
    if not found:
        error_exit(
            f"ไม่พบโมเดล '{model}' ในระบบ Ollama\n"
            f"โมเดลที่มี: {', '.join(models_available)}\n"
            f"กรุณาดาวน์โหลดก่อน: ollama pull {model}"
        )

# ── Core PDF Processing Logic ─────────────────────────────────────────────────
def process_pdf(
    pdf_path: Path,
    model: str = DEFAULT_MODEL,
    host: str = OLLAMA_HOST,
    dpi: int = DEFAULT_DPI,
) -> dict[str, Any]:
    """สกัดข้อมูลจาก PDF โดยเลือกวิธีที่เร็วที่สุดก่อน (Text -> Image Fallback)"""
    check_ollama(host, model)
    
    # 1. ลองดึงข้อมูลด้วย Text Extraction (Fast Path)
    extracted_text = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                extracted_text += t + "\n"
    except Exception as e:
        warn(f"ไม่สามารถดึงข้อความดิจิทัลได้: {e}")

    extracted_text = extracted_text.strip()
    
    # ตรวจสอบว่ามีตัวอักษรมากพอหรือไม่ หากน้อยเกินไปให้สันนิษฐานว่าเป็นรูปสแกน
    if len(extracted_text) > 50:
        info("ตรวจพบข้อมูลแบบ Text — กำลังประมวลผลด่วนด้วย Text Mode...")
        return process_text_mode(extracted_text, model, host)
    
    # 2. ทำ Image-based OCR ด้วย Vision Model (Slow Fallback Path)
    info("ไม่พบข้อมูลแบบ Text (อาจเป็นไฟล์สแกน) — กำลังประมวลผลด้วย Image Mode (Vision)...")
    return process_image_mode(pdf_path, model, host, dpi)

def process_text_mode(text: str, model: str, host: str) -> dict[str, Any]:
    """ส่งข้อความที่ดึงได้จาก PDF ไปประมวลผลเป็น JSON"""
    prompt = textwrap.dedent(f"""
        You are an expert invoice OCR and parser system. Analyze the following extracted text from an invoice and extract the target fields.
        
        To ensure high accuracy, use a Chain-of-Thought approach. In your JSON response:
        1. First, under the "thinking_steps" key, explain step-by-step where in the text you found the values for each target field.
        2. Then, populate the extracted fields.

        Invoice Text Content:
        \"\"\"
        {text}
        \"\"\"

        Return ONLY a valid JSON object matching the JSON schema below.
        DO NOT include markdown block tags like ```json or any explanations outside the JSON.
        
        JSON Schema:
        {{
          "thinking_steps": "string explaining how you located each field step-by-step",
          "tax_provider": "string or null",
          "tax_merchant": "string or null",
          "date": "string or null",
          "document_id": "string or null",
          "merchant_address": "string or null",
          "merchant_branch": "string or null",
          "total_discount_amount": "number or null",
          "total_value_added_tax": "number or null"
        }}
    """).strip()

    payload = {
        "model": model,
        "stream": False,
        "format": "json",  # บังคับโครงสร้าง JSON จาก Ollama
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
        },
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    resp = requests.post(f"{host}/api/chat", json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    raw_response = resp.json()["message"]["content"]
    return clean_and_parse_json(raw_response)

def process_image_mode(pdf_path: Path, model: str, host: str, dpi: int) -> dict[str, Any]:
    """แปลงหน้าแรกของ PDF เป็นรูปภาพและใช้ Vision Model สกัดข้อมูล"""
    try:
        images = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as e:
        raise RuntimeError(f"แปลง PDF เป็นรูปภาพล้มเหลว (ตรวจสอบว่าติดตั้ง poppler หรือยัง): {e}")

    if not images:
        raise RuntimeError("ไม่พบหน้าในไฟล์ PDF นี้")

    # ประมวลผลเฉพาะหน้าแรก (ปกติเอกสารสำคัญหรือ Invoice สรุปข้อมูลอยู่ที่หน้าแรก)
    image = images[0]
    w, h = image.size
    if max(w, h) > MAX_IMAGE_PX:
        scale = MAX_IMAGE_PX / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = textwrap.dedent("""
        You are an expert invoice OCR and parser system. Analyze this invoice image and extract the target fields.
        
        To maximize OCR accuracy, you MUST use a Chain-of-Thought (CoT) transcription approach:
        1. First, under the "ocr_transcription" key, perform a detailed line-by-line transcription of the invoice text, including all names, numbers, dates, addresses, and tables. Write down exactly what you see.
        2. Second, under the "thinking_steps" key, explain step-by-step where in the "ocr_transcription" you found the values for each target field.
        3. Finally, populate the extracted fields with the correct values.

        Return ONLY a valid JSON object matching the JSON schema below.
        DO NOT include markdown block tags like ```json or any explanations outside the JSON.

        JSON Schema:
        {
          "ocr_transcription": "detailed line-by-line transcription of the invoice image",
          "thinking_steps": "step-by-step reasoning explaining how you located each field in the transcription",
          "tax_provider": "string or null",
          "tax_merchant": "string or null",
          "date": "string or null",
          "document_id": "string or null",
          "merchant_address": "string or null",
          "merchant_branch": "string or null",
          "total_discount_amount": "number or null",
          "total_value_added_tax": "number or null"
        }
    """).strip()

    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
        },
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [img_b64]
            }
        ]
    }

    resp = requests.post(f"{host}/api/chat", json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    raw_response = resp.json()["message"]["content"]
    return clean_and_parse_json(raw_response)

# ── FastAPI Application Setup ─────────────────────────────────────────────────
api_app = FastAPI(
    title="Invoice OCR API",
    description="API สำหรับสกัดข้อมูลจาก PDF Invoice ส่งออกเป็นโครงสร้างข้อมูล JSON",
    version="1.0.0"
)

# เพิ่ม CORS Middleware ให้ดึงไปใช้จากหน้าเว็บอื่นได้
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api_app.post("/api/extract", summary="สกัดข้อมูลจากไฟล์ PDF Invoice")
def api_extract(
    file: UploadFile = File(..., description="ไฟล์ PDF Invoice ที่ต้องการประมวลผล"),
    model: str = DEFAULT_MODEL,
    host: str = OLLAMA_HOST
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="ต้องเป็นไฟล์นามสกุล .pdf เท่านั้น")
    
    # บันทึกไฟล์อัปโหลดชั่วคราวเพื่อนำไปประมวลผล
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / file.filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = process_pdf(temp_path, model=model, host=host)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # ลบไฟล์ชั่วคราวทิ้งเสมอหลังสแกนเสร็จ
        if temp_path.exists():
            temp_path.unlink()

# ── CLI Entry Point ───────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf_extractor",
        description="สกัดข้อมูลสำคัญจาก PDF Invoice เป็นโครงสร้าง JSON ด้วย Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        ตัวอย่างการใช้งานแบบ CLI:
          python pdf.py invoice.pdf
          python pdf.py invoice.pdf --model gemma3:4b
          
        ตัวอย่างการเปิดเว็บบอร์ด API Server:
          python pdf.py --server
          python pdf.py --server --port 8000
        """),
    )
    parser.add_argument("pdf", metavar="PDF_FILE", nargs="?", default=None,
                        help="ไฟล์ PDF ที่ต้องการสแกน (หากเว้นว่างไว้ จะแสดงหน้าต่างให้เลือกไฟล์)")
    parser.add_argument("--server", action="store_true",
                        help="รันเป็น API Server (FastAPI)")
    parser.add_argument("--host", metavar="URL", default=OLLAMA_HOST,
                        help=f"Ollama API URL (default: {OLLAMA_HOST})")
    parser.add_argument("--model", metavar="NAME", default=DEFAULT_MODEL,
                        help=f"ชื่อโมเดล Ollama (default: {DEFAULT_MODEL})")
    parser.add_argument("--port", type=int, default=8000,
                        help="พอร์ตสำหรับรัน API Server (default: 8000)")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                        help=f"ความละเอียดรูปภาพสำหรับไฟล์สแกน (default: {DEFAULT_DPI})")
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # กรณี 1: รันเป็น API Server
    if args.server:
        if not HAS_WEB_API:
            error_exit("ไม่สามารถเปิดโหมด Server ได้ เนื่องจากขาดแพ็คเกจ 'fastapi' หรือ 'uvicorn'")
        
        info(f"เริ่มต้น API Server ที่ http://localhost:{args.port}")
        info(f"API Endpoint: [bold green]POST http://localhost:{args.port}/api/extract[/bold green]")
        info(f"ลองส่งไฟล์ตรวจผ่าน Swagger UI ที่: http://localhost:{args.port}/docs")
        
        uvicorn.run(api_app, host="0.0.0.0", port=args.port)
        return

    # กรณี 2: รันแบบ CLI
    pdf_path = None
    if args.pdf:
        pdf_path = Path(args.pdf)
    else:
        # เปิด GUI File Picker
        info("ไม่พบไฟล์ PDF ในคำสั่ง, กำลังเปิดกล่องเลือกไฟล์...")
        pdf_path = select_file_gui()
        if not pdf_path:
            error_exit("ไม่ได้เลือกไฟล์ PDF")

    if not pdf_path.exists():
        error_exit(f"ไม่พบไฟล์: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        error_exit(f"ต้องเป็นไฟล์ .pdf แต่ได้: {pdf_path.suffix}")

    if HAS_RICH:
        t = Table.grid(padding=(0, 2))
        t.add_column(style="bold cyan")
        t.add_column()
        t.add_row("📑  ไฟล์", pdf_path.name)
        t.add_row("🤖  โมเดล", args.model)
        console.print(Panel(t, title="[bold]Invoice Parser[/bold]", border_style="cyan", expand=False))
    else:
        print(f"--- Invoice Parser ---\n📑 ไฟล์: {pdf_path.name}\n🤖 โมเดล: {args.model}\n----------------------")

    try:
        start_time = time.time()
        result = process_pdf(pdf_path, model=args.model, host=args.host, dpi=args.dpi)
        elapsed = time.time() - start_time

        success(f"ดำเนินการเสร็จสิ้นในเวลา {elapsed:.1f} วินาที\n")
        
        # แสดงผลลัพธ์เป็นโครงสร้าง JSON แบบสวยงาม
        json_output = json.dumps(result, indent=2, ensure_ascii=False)
        if HAS_RICH:
            console.print(Syntax(json_output, "json", theme="github-dark"))
        else:
            print(json_output)
            
    except Exception as e:
        error_exit(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")

if __name__ == "__main__":
    main()