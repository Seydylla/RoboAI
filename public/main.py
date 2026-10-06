import sys
import os
import sqlite3
import threading
import warnings
import io
import re
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

# PyTorch & Hugging Face Transformers for local TTS
import torch
from transformers import VitsModel, AutoTokenizer

# Matplotlib for visual slide graphs
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Qt
import matplotlib.pyplot as plt
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QRect
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGraphicsDropShadowEffect, QPushButton
)
from PyQt6.QtGui import QColor, QFont, QPalette, QLinearGradient, QBrush, QPixmap, QImage, QPainter, QPen

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
# Worker Thread: Generates Lesson, Topic Graphs, and TTS Audio
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
            target_word_count = max(600, int(self.duration_minutes * 600))

            prompt = f"""
Siz tejribeli mugallym. Sapagyň mowzugy: {self.subject} - {self.topic}.
Sapagyň dowamlylygy: {self.duration_minutes} minut.

Haýyş, ähli jogaby diňe Türkmen dilinde (Latyn elipbiýinde) doly, giňišleýin we düşnükli beriň.

Wajyp düzgünler:
1. Sapagy örän gyzykly, özüne çekiji, janly we TÄSIRLI ediň, hiç hili içgysgyn bolmasyn! (Make really interesting rather than boring).
2. Sapagyň dowamlylygy {self.duration_minutes} minut bolany üçin hut {target_slide_count} sany slayd dörediň (her minut üçin 2 slayd).
3. FULL_SPEECH bölüminde edil {target_word_count} söz töweregi (her minut üçin 600 söz) giňišleýin düşündiriş ýazyň.
4. Gürrüňiň içinde minutlary asla agzamaň (meselem: "häzir 1-nji minutda", "2-nji minutdarys", "minut geçdi" diýip AÝTMAŇ!).
5. Riyazi formulalarda we simwollarda raw LaTeX ($...$, \\cdot, ^{{n}}) ulanmaň! Onuň deregine ýönekeý Unicode simwollaryny ulanyň (x², f'(x), xⁿ, ·, ±, ∫, √, π, ≤, ≥).

Jogaby tapawutlandyrmak üçin edil ashakdaky ýaly strukturada ýazyň:

SLIDE_1:
Sözbaşy: [1-nji Slaydyň gysga sözbaşysy]
Mazmuny:
- [Täsirli we düşnükli esasy nokat]
- [Eminlik bilen düşündirilýän ikinji nokat]

SLIDE_2:
Sözbaşy: [2-nji Slaydyň sözbaşysy]
Mazmuny:
- [Düşündirişler we mysallar]

FULL_SPEECH:
[Bu ýerde çagalara aýtjak takmynan {target_word_count} sözden ybarat bolan, örän gyzykly, janly gürrüňiňizi ýazyň. Minutlary sanamaň!]
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

            self.status_signal.emit("Slaydlar we grafikler döredilýär...")
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
            for line in lines:
                if line.startswith("Sözbaşy:"):
                    title = clean_latex_math(line.replace("Sözbaşy:", "").strip())
                elif line.startswith("Mazmuny:"):
                    continue
                else:
                    content_lines.append(clean_latex_math(line))

            slides.append({
                "title": title, 
                "content": "\n".join(content_lines)
            })

        if not slides:
            slides = [{"title": self.subject, "content": self.topic}]

        return {"slides": slides, "speech": clean_latex_math(speech)}

    def generate_topic_graph(self, slide_index, slide_title=""):
        """Generates dynamic topic-specific dark-themed plots using Matplotlib."""
        fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=100)
        fig.patch.set_facecolor('#1e293b')
        ax.set_facecolor('#0f172a')

        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_color('#334155')
        ax.spines['right'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.tick_params(axis='x', colors='#94a3b8')
        ax.tick_params(axis='y', colors='#94a3b8')
        ax.title.set_color('#f8fafc')

        full_topic = (f"{self.topic} {self.subject} {slide_title}").lower()

        # 1. Basic Arithmetic / Addition / Subtraction / Numbers
        if any(w in full_topic for w in ["goşmak", "aýyrmak", "kópleltmek", "bölmek", "addition", "subtraction", "sum", "plus"]):
            if slide_index % 2 == 0:
                ax.axhline(0, color='#94a3b8', linewidth=2)
                ax.plot([0, 2], [0, 0.5], color='#38bdf8', linewidth=3, label='+2')
                ax.plot([2, 5], [0.5, 0], color='#4ade80', linewidth=3, label='+3')
                ax.scatter([0, 2, 5], [0, 0.5, 0], color='#f8fafc', s=60, zorder=5)
                ax.set_xlim(-1, 7)
                ax.set_ylim(-0.5, 1)
                ax.set_yticks([])
                ax.set_xticks(range(0, 7))
                ax.set_title("San Okunda Goşmak (2 + 3 = 5)", fontsize=10)
                ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', loc='upper right')
            else:
                labels = ['San 1', 'San 2', 'Jemi (Sum)']
                values = [2, 3, 5]
                colors = ['#38bdf8', '#a855f7', '#4ade80']
                bars = ax.bar(labels, values, color=colors, width=0.5)
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, int(yval), ha='center', va='bottom', color='#f8fafc', fontweight='bold')
                ax.set_ylim(0, 7)
                ax.set_title("Goşulyjylar we Jemi", fontsize=10)
            ax.grid(axis='y', color='#334155', linestyle=':', alpha=0.6)

        # 2. Calculus / Derivatives / Integrals
        elif any(w in full_topic for w in ["kalkulus", "calculus", "önüm", "töreme", "derivative", "integral"]):
            x = np.linspace(-3, 3, 200)
            if slide_index % 2 == 0:
                y1 = x**2
                y2 = 2*x
                ax.plot(x, y1, color='#38bdf8', linewidth=2.5, label='f(x) = x²')
                ax.plot(x, y2, color='#a855f7', linewidth=2, linestyle='--', label="f'(x) = 2x")
                ax.set_title("Funksiýa we Onuň Önümi", fontsize=10)
            else:
                y1 = x**3 - 3*x
                ax.plot(x, y1, color='#f43f5e', linewidth=2.5, label='f(x) = x³ - 3x')
                ax.set_title("Kübiki Funksiýa", fontsize=10)
            ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', loc='upper left')
            ax.grid(True, color='#334155', linestyle=':', alpha=0.6)

        # 3. Trigonometry
        elif any(w in full_topic for w in ["trigonometriýa", "trigonometry", "sin", "cos", "tan", "burç"]):
            x = np.linspace(-3, 3, 200)
            y1 = np.sin(x)
            y2 = np.cos(x)
            ax.plot(x, y1, color='#4ade80', linewidth=2.5, label='sin(x)')
            ax.plot(x, y2, color='#f43f5e', linewidth=2, linestyle='--', label='cos(x)')
            ax.set_title("Trigonometriýa Grafigi", fontsize=10)
            ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', loc='upper left')
            ax.grid(True, color='#334155', linestyle=':', alpha=0.6)

        # 4. Computer Science / Programming / IT
        elif any(w in full_topic for w in ["programlama", "it", "code", "web", "programming", "python", "php", "js"]):
            categories = ['HTML', 'CSS', 'JS', 'PHP', 'Python']
            values = [85, 90, 75, 95, 80]
            ax.bar(categories, values, color=['#38bdf8', '#facc15', '#a855f7', '#4ade80', '#f43f5e'])
            ax.set_title("Programmalaşdyryş Bilişi (%)", fontsize=10)
            ax.grid(axis='y', color='#334155', linestyle=':', alpha=0.6)

        # 5. Default General Learning Curve Plot
        else:
            x = np.linspace(1, 10, 100)
            y = np.log(x) * 10
            ax.plot(x, y, color='#a855f7', linewidth=2.5)
            ax.fill_between(x, y, color='#a855f7', alpha=0.2)
            ax.set_title("Sapak Boýunça Bilim Ösüşi", fontsize=10)
            ax.grid(True, color='#334155', linestyle=':', alpha=0.6)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor(), transparent=False)
        plt.close(fig)
        buf.seek(0)

        image = QImage()
        image.loadFromData(buf.getvalue())
        return image

    def render_slides_to_pixmaps(self, slides):
        """Generates slide graphics with wrapped text, clean math chars, and context-matching graphs."""
        pixmaps = []
        for idx, slide in enumerate(slides):
            img = QImage(1280, 720, QImage.Format.Format_ARGB32)
            img.fill(QColor("#0f172a"))

            painter = QPainter(img)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(30, 41, 59, 240)))
            painter.drawRoundedRect(40, 40, 1200, 640, 20, 20)

            painter.setBrush(QBrush(QColor(56, 189, 248)))
            painter.drawRoundedRect(70, 70, 10, 45, 5, 5)

            painter.setPen(QColor(248, 250, 252))
            painter.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
            title_rect = QRect(95, 68, 1100, 50)
            painter.drawText(title_rect, Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, slide["title"])

            painter.setPen(QColor(203, 213, 225))
            painter.setFont(QFont("Segoe UI", 16))
            content_rect = QRect(95, 145, 650, 500)
            painter.drawText(content_rect, Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, slide["content"])

            graph_img = self.generate_topic_graph(idx, slide["title"])
            if not graph_img.isNull():
                painter.drawImage(760, 150, graph_img)

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
        main_layout.setContentsMargins(40, 30, 40, 30)

        card = QWidget(self)
        card.setObjectName("GlassCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(30, 20, 30, 20)
        card_layout.setSpacing(10)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(50)
        shadow.setColor(QColor(0, 0, 0, 140))
        shadow.setOffset(0, 10)
        card.setGraphicsEffect(shadow)

        self.subject_label = QLabel(self.subject)
        self.subject_label.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        self.subject_label.setStyleSheet("color: #ffffff; background: transparent;")
        self.subject_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.subject_label)

        self.topic_label = QLabel(f"Mowzuk: {self.topic}")
        self.topic_label.setFont(QFont("Segoe UI", 14))
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
                border-radius: 16px;
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
        self.slide_counter.setStyleSheet("color: #94a3b8; font-size: 15px; font-weight: bold;")
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
                background-color: rgba(255, 255, 255, 0.07);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 24px;
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
        scaled = pixmap.scaled(
            self.slide_display.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.slide_display.setPixmap(scaled)
        self.slide_counter.setText(f"Slaýd {self.current_slide_idx + 1} / {len(self.slide_pixmaps)}")

        # Auto slide timing calculation
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