import sys
import os
import sqlite3
import threading
import warnings
import io
import re
import urllib.request
import urllib.parse
import json
import textwrap
import ssl

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

# PyTorch & Hugging Face Transformers for local TTS
import torch
from transformers import VitsModel, AutoTokenizer

# Matplotlib for visual slide graphics
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Qt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QRect, QRectF
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGraphicsDropShadowEffect, QPushButton
)
from PyQt6.QtGui import (
    QColor, QFont, QPalette, QLinearGradient, QBrush, 
    QPixmap, QImage, QPainter, QPen, QPainterPath
)

# Gemini API Client
from google import genai
from google.genai import types

# Audio playback
import sounddevice as sd


# -------------------------------------------------------------------
# Local TTS Lazy Loader & Model Cache
# -------------------------------------------------------------------
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
    """Fetches high-quality educational photos/maps/diagrams from Wikimedia Commons and Wikipedia."""
    if not query:
        return None
    try:
        clean_q = re.sub(r'[^a-zA-Z0-9\s]', '', query).strip()
        if not clean_q:
            return None

        # Bypass SSL verification issues on local Python environments
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        headers = {
            'User-Agent': 'EducationalLessonApp/1.0 (student_learning_app@example.org)'
        }

        # 1. Search Wikipedia Page Images (Most reliable for historical topics/WW2/Science)
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

        # 2. Search Wikimedia Commons Media Files
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
        r'\cdot': '·',
        r'\times': '×',
        r'\div': '÷',
        r'\pm': '±',
        r'\infty': '∞',
        r'\pi': 'π',
        r'\alpha': 'α',
        r'\beta': 'β',
        r'\theta': 'θ',
        r'\le': '≤',
        r'\leq': '≤',
        r'\ge': '≥',
        r'\geq': '≥',
        r'\neq': '≠',
        r'\approx': '≈',
        r'\sqrt': '√',
        r'\int': '∫',
        r'\sum': '∑',
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


# -------------------------------------------------------------------
# Worker Thread: Generates Lesson, Visuals, and TTS Audio
# -------------------------------------------------------------------
class LessonGeneratorThread(QThread):
    status_signal = pyqtSignal(str)
    completed_signal = pyqtSignal(dict, list, object)
    error_signal = pyqtSignal(str)

    def __init__(self, subject, topic, duration_minutes, api_key):
        super().__init__()
        self.subject = subject
        self.topic = topic
        self.duration_minutes = duration_minutes
        self.api_key = api_key

    def run(self):
        try:
            if not self.api_key or "YOUR_GEMINI_API_KEY" in self.api_key:
                raise ValueError("Gemini API key tapylmady!")

            self.status_signal.emit("Gemini AI sapagy taýýarlaýar...")
            client = genai.Client(api_key=self.api_key)

            target_slide_count = max(2, int(self.duration_minutes * 2))
            target_word_count = max(300, int(self.duration_minutes * 300))

            prompt = f"""
Siz tejribeli mugallym. Sapagyň mowzugy: {self.subject} - {self.topic}.
Sapagyň dowamlylygy: {self.duration_minutes} minut.

Haýyş, ähli jogaby diňe Türkmen dilinde (Latyn elipbiýinde) doly, giňišleýin we düşnükli beriň.

Wajyp düzgünler:
1. Sapagy örän gyzykly, özüne çekiji, janly we TÄSIRLI ediň!
2. Sapagyň dowamlylygy {self.duration_minutes} minut bolany üçin hut {target_slide_count} sany slayd dörediň.
3. FULL_SPEECH bölüminde edil {target_word_count} söz töweregi giňišleýin düşündiriş ýazyň. Gürrüňiň içinde minutlary asla agzamaň.
4. Her slayd üçin `IMAGE_QUERY` bölüminde real taryhy surat, karta ýa-da illustrasiýa tapmak üçin diňe IŇLISÇE 2-3 sany giňden belli açar sözüni beriň (meselem: "World War 2 Europe map", "Battle of Stalingrad", "Battle of Kursk").
5. `GRAPH_CODE` diňe matematika we fizika ýaly takyk ylymlar üçin Matplotlib kody bolsun. Taryh, edebiýat, geografiýa ýaly derslerde GRAPH_CODE-y boş goýuň.
6. Syýasy tarap tutmaly däl. Temany garaşzys şekilde okatmaly.
7. Riyazi formulalarda raw LaTeX ulanmaň, ýönekeý Unicode simwollaryny ulanyň.

Jogaby tapawutlandyrmak üçin edil aşakdaky ýaly strukturada ýazyň:

SLIDE_1:
Sözbaşy: [1-nji Slaydyň gysga sözbaşysy]
Mazmuny:
- [Tema we öwrediljek zatlara degişli 80 we 170 aralygynda söz]
IMAGE_QUERY: [2-3 English Wikipedia search keywords]
GRAPH_CODE:
[Diňe Python matplotlib ax kody]

SLIDE_2:
Sözbaşy: [2-nji Slaydyň sözbaşysy]
Mazmuny:
- [Tema we öwrediljek zatlara degişli 80 we 170 aralygynda söz]
IMAGE_QUERY: [2-3 English Wikipedia search keywords]
GRAPH_CODE:
[Diňe Python matplotlib ax kody]

FULL_SPEECH:
[Bu ýerde çagalara aýtjak takmynan {target_word_count} sözden ybarat bolan gürrüňiňizi ýazyň.]
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
            parsed_data = self.parse_gemini_output(raw_text)

            self.status_signal.emit("Slaydlar we internet suratlary ýüklenýär...")
            slide_pixmaps = self.render_slides_to_pixmaps(parsed_data["slides"])

            self.status_signal.emit("Ses emele getirilýär (TTS 1.2x)...")
            audio_data, sample_rate = self.generate_tts_local(parsed_data["speech"], speed_factor=1.2)

            self.completed_signal.emit(parsed_data, slide_pixmaps, (audio_data, sample_rate))

        except Exception as e:
            self.error_signal.emit(str(e))

    def parse_gemini_output(self, text):
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
                "title": self.subject, 
                "content": self.topic, 
                "image_query": f"{self.subject} {self.topic}",
                "graph_code": ""
            }]

        return {"slides": slides, "speech": clean_latex_math(speech)}

    def generate_topic_graph(self, graph_code="", slide_title=""):
        """Executes AI plot code or creates clean infographic visual without naked coordinate axes."""
        fig, ax = plt.subplots(figsize=(5.4, 5.5), dpi=100)
        fig.patch.set_facecolor('#1e293b')
        ax.set_facecolor('#0f172a')

        clean_code = re.sub(r'```python|```', '', graph_code).strip()

        if clean_code:
            local_scope = {
                'ax': ax, 
                'np': np, 
                'plt': plt, 
                'patches': mpatches, 
                'mpatches': mpatches
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
            # Clean stylized infographic block instead of coordinate grid
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

    def render_slides_to_pixmaps(self, slides):
        """Generates full-card visual slides with online images or clean diagrams."""
        pixmaps = []
        for slide in slides:
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
            
            # Fetch real online image from Wikipedia / Wikimedia Commons
            qimg = None
            if slide.get("image_query"):
                qimg = fetch_online_image(slide["image_query"])
            if qimg is None and slide.get("title"):
                qimg = fetch_online_image(f"{self.topic} {slide['title']}")
            if qimg is None:
                qimg = fetch_online_image(self.topic)

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
                # Render clean diagram / plot if no internet photo was found
                graph_img = self.generate_topic_graph(slide.get("graph_code", ""), slide["title"])
                if not graph_img.isNull():
                    painter.drawImage(visual_rect.x(), visual_rect.y(), graph_img)

            painter.end()
            pixmaps.append(QPixmap.fromImage(img))

        return pixmaps

    def generate_tts_local(self, text, speed_factor=1.2):
        """Generates full speech using chunking at 1.2x speed."""
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
            print("Local TTS Error:", e)
            return None, None


# -------------------------------------------------------------------
# Main App Window
# -------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(self.base_dir, "database", "lessons.db")

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        self.subject = "Matematika"
        self.topic = "Goşmak"
        self.duration_minutes = 10
        self.lesson_data = None
        self.slide_pixmaps = []
        self.current_slide_idx = 0

        self.slide_timer = QTimer(self)
        self.slide_timer.timeout.connect(self.auto_next_slide)

        self.load_latest_lesson()
        self.init_ui()
        self.showFullScreen()

        if self.gemini_api_key and "YOUR_GEMINI_API_KEY" not in self.gemini_api_key:
            self.start_ai_generation()
        else:
            self.status_label.setText("Ýalňyşlyk: GEMINI_API_KEY girizilmedik!")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.slide_pixmaps:
            self.display_current_slide()

    def load_latest_lesson(self):
        if not os.path.exists(self.db_path):
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT subject, topic, duration_minutes FROM lessons ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            if row:
                self.subject, self.topic, self.duration_minutes = row[0], row[1], row[2]
        except sqlite3.Error as e:
            print("Database error:", e)

    def init_ui(self):
        self.setWindowTitle("AI Mugallym - Sapak")
        self.setAutoFillBackground(True)
        palette = self.palette()
        gradient = QLinearGradient(0, 0, 1920, 1080)
        gradient.setColorAt(0.0, QColor("#0f172a"))
        gradient.setColorAt(0.5, QColor("#1e1b4b"))
        gradient.setColorAt(1.0, QColor("#311042"))
        palette.setBrush(QPalette.ColorRole.Window, QBrush(gradient))
        self.setPalette(palette)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 10, 15, 10)

        card = QWidget(self)
        card.setObjectName("GlassCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(15, 10, 15, 10)
        card_layout.setSpacing(6)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(50)
        shadow.setColor(QColor(0, 0, 0, 140))
        shadow.setOffset(0, 10)
        card.setGraphicsEffect(shadow)

        self.subject_label = QLabel(self.subject)
        self.subject_label.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        self.subject_label.setStyleSheet("color: #ffffff; background: transparent;")
        self.subject_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.subject_label)

        self.topic_label = QLabel(f"Mowzuk: {self.topic}")
        self.topic_label.setFont(QFont("Segoe UI", 13))
        self.topic_label.setStyleSheet("color: #a855f7; background: transparent; font-weight: 600;")
        self.topic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.topic_label)

        self.status_label = QLabel("Sapak ýüklenýär...")
        self.status_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.status_label.setStyleSheet("color: #38bdf8; background: transparent;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.status_label)

        # Slide Display Screen Zone
        self.slide_display = QLabel()
        self.slide_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slide_display.setStyleSheet("""
            QLabel {
                background-color: rgba(15, 23, 42, 0.6);
                border: none;
                border-radius: 12px;
            }
        """)
        card_layout.addWidget(self.slide_display, stretch=1)

        # Slide Navigation Controls
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("◀ Yza")
        self.prev_btn.setFixedWidth(120)
        self.prev_btn.setStyleSheet(self.btn_style("#3b82f6"))
        self.prev_btn.clicked.connect(self.prev_slide)
        
        self.slide_counter = QLabel("Slaýd 0 / 0")
        self.slide_counter.setStyleSheet("color: #94a3b8; font-size: 14px; font-weight: bold;")
        self.slide_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.next_btn = QPushButton("Öňe ▶")
        self.next_btn.setFixedWidth(120)
        self.next_btn.setStyleSheet(self.btn_style("#3b82f6"))
        self.next_btn.clicked.connect(self.next_slide)

        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.slide_counter)
        nav_layout.addWidget(self.next_btn)
        card_layout.addLayout(nav_layout)

        # Exit Button
        exit_btn = QPushButton("Çykmak (Esc)")
        exit_btn.setFixedWidth(140)
        exit_btn.setStyleSheet(self.btn_style("#ef4444"))
        exit_btn.clicked.connect(self.close)
        card_layout.addWidget(exit_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        main_layout.addWidget(card)
        self.setLayout(main_layout)

        self.setStyleSheet("""
            QWidget#GlassCard {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 20px;
            }
        """)

    def btn_style(self, color_hex):
        return f"""
            QPushButton {{
                background-color: {color_hex}44;
                border: 1px solid {color_hex};
                color: white;
                border-radius: 10px;
                padding: 6px 14px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {color_hex};
            }}
        """

    def start_ai_generation(self):
        self.thread = LessonGeneratorThread(
            self.subject, self.topic, self.duration_minutes, self.gemini_api_key
        )
        self.thread.status_signal.connect(self.update_status)
        self.thread.completed_signal.connect(self.on_generation_complete)
        self.thread.error_signal.connect(self.on_generation_error)
        self.thread.start()

    def update_status(self, message):
        self.status_label.setText(message)

    def on_generation_error(self, err_msg):
        self.status_label.setText(f"Ýalňyşlyk: {err_msg}")
        self.status_label.setStyleSheet("color: #f87171; background: transparent;")

    def on_generation_complete(self, lesson_data, slide_pixmaps, audio_tuple):
        self.status_label.setText("Sapak başlandy!")
        self.status_label.setStyleSheet("color: #4ade80; background: transparent;")

        self.lesson_data = lesson_data
        self.slide_pixmaps = slide_pixmaps
        self.current_slide_idx = 0
        self.display_current_slide()

        audio_data, sample_rate = audio_tuple
        if audio_data is not None and sample_rate is not None:
            threading.Thread(target=self.play_audio, args=(audio_data, sample_rate), daemon=True).start()

    def display_current_slide(self):
        if not self.slide_pixmaps:
            return
        
        pixmap = self.slide_pixmaps[self.current_slide_idx]
        target_size = self.slide_display.size()
        if target_size.width() < 100 or target_size.height() < 100:
            target_size = QSize(1280, 720)

        scaled = pixmap.scaled(
            target_size, 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.slide_display.setPixmap(scaled)
        self.slide_counter.setText(f"Slaýd {self.current_slide_idx + 1} / {len(self.slide_pixmaps)}")

        slide_count = max(1, len(self.slide_pixmaps))
        slide_duration_ms = int((self.duration_minutes * 60 * 1000) / slide_count)
        self.slide_timer.start(slide_duration_ms)

    def auto_next_slide(self):
        if self.slide_pixmaps and self.current_slide_idx < len(self.slide_pixmaps) - 1:
            self.current_slide_idx += 1
            self.display_current_slide()
        else:
            self.slide_timer.stop()
            self.status_label.setText("Sapak tamamlandy!")
            self.status_label.setStyleSheet("color: #a855f7; background: transparent;")

    def next_slide(self):
        if self.slide_pixmaps and self.current_slide_idx < len(self.slide_pixmaps) - 1:
            self.current_slide_idx += 1
            self.display_current_slide()

    def prev_slide(self):
        if self.slide_pixmaps and self.current_slide_idx > 0:
            self.current_slide_idx -= 1
            self.display_current_slide()

    def play_audio(self, audio_data, sample_rate):
        try:
            sd.play(audio_data, sample_rate)
            sd.wait()
        except Exception as e:
            print("Audio error:", e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())