import sys
import os
import sqlite3
import json
import re
import io
import ssl
import urllib.request
import urllib.parse
import wave
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

# PyTorch & Hugging Face Transformers for local TTS
import torch
from transformers import VitsModel, AutoTokenizer

# Gemini API Client
from google import genai
from google.genai import types

# PyQt6 for rendering slide graphics
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QColor, QFont, QBrush, QImage, QPainter, QPen, QPainterPath, QPixmap
from PyQt6.QtCore import Qt, QRect, QRectF

# Base directory resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(BASE_DIR, "database")):
    BASE_DIR = os.path.dirname(BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

LOCAL_TTS_MODEL = None
LOCAL_TTS_TOKENIZER = None

def get_local_tts():
    """Loads and caches the local Hugging Face TTS model in memory."""
    global LOCAL_TTS_MODEL, LOCAL_TTS_TOKENIZER
    if LOCAL_TTS_MODEL is None or LOCAL_TTS_TOKENIZER is None:
        model_name = "facebook/mms-tts-tuk-script_latin"
        LOCAL_TTS_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        LOCAL_TTS_MODEL = VitsModel.from_pretrained(model_name)
    return LOCAL_TTS_MODEL, LOCAL_TTS_TOKENIZER


def fetch_online_image(query):
    """Fetches high-quality educational photos/maps/diagrams from Wikipedia and Wikimedia Commons."""
    if not query:
        return None
    try:
        clean_q = re.sub(r'[^a-zA-Z0-9\s]', '', query).strip()
        if not clean_q:
            return None

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        headers = {
            'User-Agent': 'EducationalLessonApp/1.0 (student_learning_app@example.org)'
        }

        # Search Wikipedia Page Images
        url_wiki = f"https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch={urllib.parse.quote(clean_q)}&gsrlimit=5&prop=pageimages&pithumbsize=1000&format=json"
        req_wiki = urllib.request.Request(url_wiki, headers=headers)
        
        with urllib.request.urlopen(req_wiki, timeout=5, context=ctx) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_info in pages.items():
                if 'thumbnail' in page_info:
                    img_url = page_info['thumbnail']['source']
                    img_req = urllib.request.Request(img_url, headers=headers)
                    with urllib.request.urlopen(img_req, timeout=5, context=ctx) as img_resp:
                        qimg = QImage()
                        if qimg.loadFromData(img_resp.read()):
                            return qimg

        # Search Wikimedia Commons
        url_commons = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrnamespace=6&gsrsearch={urllib.parse.quote(clean_q)}&gsrlimit=5&prop=imageinfo&iiprop=url&iiurlwidth=1000&format=json"
        req_commons = urllib.request.Request(url_commons, headers=headers)
        
        with urllib.request.urlopen(req_commons, timeout=5, context=ctx) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_info in pages.items():
                imageinfo = page_info.get('imageinfo', [])
                if imageinfo:
                    img_url = imageinfo[0].get('thumburl') or imageinfo[0].get('url')
                    if img_url and not img_url.endswith('.svg'):
                        img_req = urllib.request.Request(img_url, headers=headers)
                        with urllib.request.urlopen(img_req, timeout=5, context=ctx) as img_resp:
                            qimg = QImage()
                            if qimg.loadFromData(img_resp.read()):
                                return qimg
    except Exception as e:
        print(f"Online image fetch notice for '{query}':", e)
    return None


def clean_latex_math(text):
    """Converts raw LaTeX math syntax into clean Unicode characters for PyQt rendering."""
    if not text:
        return ""
    
    replacements = {
        r'\cdot': '·', r'\times': '×', r'\div': '÷', r'\pm': '±',
        r'\infty': '∞', r'\pi': 'π', r'\alpha': 'α', r'\beta': 'β',
        r'\theta': 'θ', r'\le': '≤', r'\leq': '≤', r'\ge': '≥',
        r'\geq': '≥', r'\neq': '≠', r'\approx': '≈', r'\sqrt': '√',
        r'\int': '∫', r'\sum': '∑',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
        
    sup_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
        'n': 'ⁿ', 'i': 'ⁱ', 'x': 'ˣ', 'k': 'ᵏ'
    }
    
    def sup_replace_brace(match):
        content = match.group(1)
        return "".join(sup_map.get(c, c) for c in content)
    text = re.sub(r'\^\{([^}]+)\}', sup_replace_brace, text)
    
    def sup_replace_single(match):
        char = match.group(1)
        return sup_map.get(char, f"^{char}")
    text = re.sub(r'\^([0-9nixk+\-])', sup_replace_single, text)
    
    text = text.replace('$', '')
    text = re.sub(r' +', ' ', text)
    return text


def synthesize_single_text_tts(text, speed_factor=1.2):
    """Generates audio array for a single text chunk with seed locking."""
    try:
        if not text:
            return None, None
        model, tokenizer = get_local_tts()
        sample_rate = model.config.sampling_rate

        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for s in sentences:
            if len(current_chunk) + len(s) < 220:
                current_chunk += " " + s
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = s
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        audio_segments = []
        for chunk in chunks:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

            inputs = tokenizer(chunk, return_tensors="pt")
            with torch.no_grad():
                output = model(**inputs).waveform
            segment = output.squeeze().cpu().numpy()
            audio_segments.append(segment)

        if not audio_segments:
            return None, None

        full_audio = np.concatenate(audio_segments)
        if len(full_audio) > 0 and speed_factor != 1.0:
            indices = np.arange(0, len(full_audio), speed_factor)
            full_audio = np.interp(indices, np.arange(len(full_audio)), full_audio).astype(np.float32)

        return full_audio, sample_rate
    except Exception as e:
        print("TTS Synthesis error:", e)
        return None, None


def generate_topic_graph(graph_code="", slide_title=""):
    fig, ax = plt.subplots(figsize=(5.4, 5.5), dpi=100)
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#0f172a')

    clean_code = re.sub(r'```python|```', '', graph_code).strip()

    if clean_code:
        local_scope = {
            'ax': ax, 'np': np, 'plt': plt, 
            'patches': mpatches, 'mpatches': mpatches
        }
        try:
            exec(clean_code, {}, local_scope)
        except Exception as e:
            print(f"Error executing AI visual code for '{slide_title}':", e)
            ax.clear()
            ax.set_facecolor('#0f172a')
            ax.axis('off')
            ax.text(0.5, 0.5, slide_title, color='#f8fafc', ha='center', va='center', fontsize=12, fontweight='bold')
    else:
        ax.axis('off')
        ax.add_patch(mpatches.FancyBboxPatch((0.1, 0.2), 0.8, 0.6, boxstyle="round,pad=0.05", ec="#38bdf8", fc="#1e293b", lw=2))
        ax.text(0.5, 0.5, slide_title or "Sapak Görseli", color='#f8fafc', ha='center', va='center', fontsize=14, fontweight='bold', wrap=True)

    try:
        plt.tight_layout()
    except Exception:
        pass

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    image = QImage()
    image.loadFromData(buf.getvalue())
    return image


def parse_gemini_output(text, subject, topic):
    slides = []
    speech = ""

    if "FULL_SPEECH:" in text:
        parts = text.split("FULL_SPEECH:")
        slide_part = parts[0]
        speech = parts[1].strip()
    else:
        slide_part = text
        speech = text

    slide_blocks = slide_part.split("SLIDE_")
    for block in slide_blocks[1:]:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        title = "Sapak"
        content_lines = []
        graph_code = ""
        image_query = ""
        in_graph_code = False

        for line in lines:
            if line.startswith("Sözbaşy:"):
                title = clean_latex_math(line.replace("Sözbaşy:", "").strip())
                in_graph_code = False
            elif line.startswith("Mazmuny:"):
                in_graph_code = False
                continue
            elif line.startswith("IMAGE_QUERY:"):
                image_query = line.replace("IMAGE_QUERY:", "").strip()
                in_graph_code = False
            elif line.startswith("GRAPH_CODE:"):
                in_graph_code = True
                continue
            else:
                if in_graph_code:
                    graph_code += line + "\n"
                else:
                    content_lines.append(clean_latex_math(line))

        slides.append({
            "title": title, 
            "content": "\n".join(content_lines),
            "image_query": image_query,
            "graph_code": graph_code.strip()
        })

    if not slides:
        slides = [{
            "title": subject, 
            "content": topic, 
            "image_query": f"{subject} {topic}",
            "graph_code": ""
        }]

    return {"slides": slides, "speech": clean_latex_math(speech)}


def main():
    app = QApplication(sys.argv)

    db_path = os.path.join(BASE_DIR, "database", "lessons.db")
    output_dir = os.path.join(BASE_DIR, "output")
    slides_dir = os.path.join(output_dir, "slides")
    audio_dir = os.path.join(output_dir, "audio")

    # Clear previous slides and audio files if they exist
    for folder in [slides_dir, audio_dir]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Notice: Could not remove old file {file_path}: {e}")

    os.makedirs(slides_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    if not os.path.exists(db_path):
        print("Database not found!")
        return

    # Fetch latest lesson from database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT subject, topic, duration_minutes FROM lessons ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if not row:
        print("No lesson record found in database.")
        return

    subject, topic, duration_minutes = row[0], row[1], row[2]
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or "YOUR_GEMINI_API_KEY" in api_key:
        print("Error: GEMINI_API_KEY not found!")
        return

    print("Gemini AI sapagy taýýarlaýar...")
    client = genai.Client(api_key=api_key)

    target_slide_count = max(2, int(duration_minutes * 2))
    target_word_count = max(300, int(duration_minutes * 300))

    prompt = f"""
Siz mekdepde sapak berýän ýeke-täk, çynlakaý we tejribeli mugallym.
Sapagyň mowzugy: {subject} - {topic}.
Sapagyň dowamlylygy: {duration_minutes} minut.

Haýyş, ähli jogaby diňe Türkmen dilinde (Latyn elipbiýinde) doly, giňišleýin we düşnükli beriň.

Wajyp düzgünler:
1. Sapagyň style-y PODKAST ýa-da IKI ADAMIN GEPLEŞIGI BOLMALY DÄL. Diňe bir mugallymyň monology, sapak düşündirişi bolsun.
2. Sapagyň dowamlylygy {duration_minutes} minut bolany üçin hut {target_slide_count} sany slayd dörediň.
3. FULL_SPEECH bölüminde edil {target_word_count} söz töweregi giňišleýin düşündiriş ýazyň. Sözleri diňe bir Mugallymyň agzyndan çykan ýaly ýazyň. Minutlary agzamaň.
4. Her slayd üçin `IMAGE_QUERY` bölüminde real taryhy surat, karta ýa-da illustrasiýa tapmak üçin diňe IŇLISÇE 2-3 sany giňden belli açar sözüni beriň.
5. `GRAPH_CODE` diňe matematika we fizika ýaly takyk ylymlar üçin Matplotlib kody bolsun. Taryh, edebiýat, geografiýa ýaly derslerde GRAPH_CODE-y boş goýuň.
6. Riyazi formulalarda raw LaTeX ulanmaň, ýönekeý Unicode simwollaryny ulanyň.

Jogaby tapawutlandyrmak üçin edil ashakdaky yaly strukturada yazyň:

SLIDE_1:
Sözbaşy: [1-nji Slaydyň gysga sözbaşysy]
Mazmuny:
- [Tema we öwrediljek zatlara degişli 50 we 100 aralygynda söz]
IMAGE_QUERY: [2-3 English Wikipedia search keywords]
GRAPH_CODE:
[Diňe Python matplotlib ax kody]

SLIDE_2:
Sözbaşy: [2-nji Slaydyň sözbaşysy]
Mazmuny:
- [Tema we öwrediljek zatlara degişli 50 we 100 aralygynda söz]
IMAGE_QUERY: [2-3 English Wikipedia search keywords]
GRAPH_CODE:
[Diňe Python matplotlib ax kody]

FULL_SPEECH:
[Bu ýerde mugallymyň mekdep okuwçylaryna aýtjak takmynan {target_word_count} sözden ybarat bolan durnukly yzygiderli monologyny ýazyň.]
"""

    response = client.models.generate_content(
        model='gemini-3.5-flash-lite',
        contents=prompt,
        config=types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            )
        )
    )
    raw_text = response.text
    parsed_data = parse_gemini_output(raw_text, subject, topic)

    print("Slaydlar we suratlary ýüklenip render edilýär...")
    slide_file_paths = []

    for i, slide in enumerate(parsed_data["slides"]):
        img = QImage(1280, 720, QImage.Format.Format_ARGB32)
        img.fill(QColor("#0f172a"))

        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Card background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(30, 41, 59, 245)))
        painter.drawRoundedRect(15, 15, 1250, 690, 16, 16)

        # Accent tag
        painter.setBrush(QBrush(QColor(56, 189, 248)))
        painter.drawRoundedRect(35, 35, 8, 42, 4, 4)

        # Slide Title
        painter.setPen(QColor(248, 250, 252))
        painter.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        title_rect = QRect(55, 32, 1180, 50)
        painter.drawText(title_rect, Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, slide["title"])

        # Text Content Area
        painter.setPen(QColor(226, 232, 240))
        painter.setFont(QFont("Segoe UI", 15))
        content_rect = QRect(55, 100, 620, 580)
        painter.drawText(content_rect, Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, slide["content"])

        # Right Panel Visual Box
        visual_rect = QRect(700, 100, 545, 580)
        
        qimg = None
        if slide.get("image_query"):
            qimg = fetch_online_image(slide["image_query"])
        if qimg is None and slide.get("title"):
            qimg = fetch_online_image(f"{topic} {slide['title']}")
        if qimg is None:
            qimg = fetch_online_image(topic)

        if qimg and not qimg.isNull():
            scaled_img = qimg.scaled(
                visual_rect.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            off_x = visual_rect.x() + (visual_rect.width() - scaled_img.width()) // 2
            off_y = visual_rect.y() + (visual_rect.height() - scaled_img.height()) // 2
            
            path = QPainterPath()
            path.addRoundedRect(QRectF(off_x, off_y, scaled_img.width(), scaled_img.height()), 12, 12)
            
            painter.save()
            painter.setClipPath(path)
            painter.drawImage(off_x, off_y, scaled_img)
            painter.restore()

            painter.setPen(QPen(QColor(51, 65, 85), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(off_x, off_y, scaled_img.width(), scaled_img.height()), 12, 12)
        else:
            graph_img = generate_topic_graph(slide.get("graph_code", ""), slide["title"])
            if not graph_img.isNull():
                painter.drawImage(visual_rect.x(), visual_rect.y(), graph_img)

        painter.end()

        slide_path = os.path.join(slides_dir, f"slide_{i}.png")
        img.save(slide_path)
        slide_file_paths.append(slide_path)

    print("Ses emele getirilýär (Mugallym Sesi)...")
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', parsed_data["speech"]) if s.strip()]
    audio_file_paths = []
    final_sample_rate = 16000

    for idx, sentence in enumerate(sentences):
        chunk_audio, sample_rate = synthesize_single_text_tts(sentence, speed_factor=1.2)
        if chunk_audio is not None:
            final_sample_rate = sample_rate
            wav_path = os.path.join(audio_dir, f"chunk_{idx}.wav")
            audio_int16 = (chunk_audio * 32767).astype(np.int16)
            
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())

            audio_file_paths.append(wav_path)

    json_output_path = os.path.join(output_dir, "lesson_data.json")
    output_meta = {
        "subject": subject,
        "topic": topic,
        "duration_minutes": duration_minutes,
        "slides": slide_file_paths,
        "audio_chunks": audio_file_paths,
        "sample_rate": final_sample_rate
    }

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(output_meta, f, ensure_ascii=False, indent=2)

    print("Generasiýa üstünlikli tamamlandy!")

if __name__ == "__main__":
    main()