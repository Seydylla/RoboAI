import sys
import os
import sqlite3
import threading
import warnings
import io
import re
import time
import json
import wave
import numpy as np

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

# PyTorch & Hugging Face Transformers for local TTS (Live Q&A)
import torch
from transformers import VitsModel, AutoTokenizer

# OpenCV & MediaPipe for Camera Gesture & Hand Detection
import cv2
import mediapipe as mp

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGraphicsDropShadowEffect, QPushButton
)
from PyQt6.QtGui import (
    QColor, QFont, QPalette, QLinearGradient, QBrush, QPixmap
)

# Gemini API Client
from google import genai
from google.genai import types

# Audio playback and mic recording
import sounddevice as sd

# -------------------------------------------------------------------
# Local TTS Lazy Loader & Model Cache for Live Q&A
# -------------------------------------------------------------------
LOCAL_TTS_MODEL = None
LOCAL_TTS_TOKENIZER = None

def get_local_tts():
    """Loads and caches the local Hugging Face TTS model in memory for live responses."""
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
    """Generates audio array for live Q&A response."""
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


# -------------------------------------------------------------------
# Background Camera Thread for Hand Gesture Detection & Quiz Mode
# -------------------------------------------------------------------
class CameraThread(QThread):
    hand_detected_signal = pyqtSignal()
    gesture_detected_signal = pyqtSignal(str, str) # Right gesture, Left gesture

    def __init__(self):
        super().__init__()
        self.running = True
        self.enabled = False
        self.quiz_mode = False

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Notice: Camera not found or inaccessible.")
            return

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        last_trigger = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.msleep(50)
                continue

            if self.enabled:
                frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if not self.quiz_mode:
                    # Lesson Q&A mode: Check if a hand is raised
                    if results.multi_hand_landmarks:
                        now = time.time()
                        if now - last_trigger > 6:
                            last_trigger = now
                            self.hand_detected_signal.emit()
                else:
                    # Quiz Gesture Recognition Mode (matching student.py rules)
                    right_gesture = "None"
                    left_gesture = "None"
                    
                    if results.multi_hand_landmarks and results.multi_handedness:
                        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                            handedness = results.multi_handedness[i].classification[0].label
                            landmarks = hand_landmarks.landmark
                            
                            def is_up(tip_idx, pip_idx):
                                return landmarks[tip_idx].y < landmarks[pip_idx].y - 0.02

                            fingers = [
                                is_up(8, 6),   # Index
                                is_up(12, 10), # Middle
                                is_up(16, 14), # Ring
                                is_up(20, 18)  # Pinky
                            ]
                            num_up = sum(fingers)
                            is_thumb_up = landmarks[4].y < landmarks[3].y - 0.03

                            if handedness == "Right":
                                if num_up == 1 and fingers[0]: right_gesture = '1'
                                elif num_up == 2 and fingers[0] and fingers[1]: right_gesture = '2'
                                elif num_up == 3 and fingers[0] and fingers[1] and fingers[2]: right_gesture = '3'
                                elif num_up == 4 and all(fingers): right_gesture = '4'
                            elif handedness == "Left":
                                if is_thumb_up and num_up == 0: left_gesture = 'accept'
                                elif num_up == 4 and all(fingers): left_gesture = 'cancel'

                    self.gesture_detected_signal.emit(right_gesture, left_gesture)

            self.msleep(100)
            
        hands.close()
        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# -------------------------------------------------------------------
# Thread-Safe Audio Playback & Student Q&A Execution
# -------------------------------------------------------------------
class PlaybackThread(QThread):
    update_status_signal = pyqtSignal(str, str)
    pause_timer_signal = pyqtSignal()
    resume_timer_signal = pyqtSignal()
    lesson_finished_signal = pyqtSignal()

    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self.running = True

    def run(self):
        while self.running and self.mw.current_chunk_idx < len(self.mw.audio_chunks):
            chunk = self.mw.audio_chunks[self.mw.current_chunk_idx]
            
            if self.mw.interrupt_requested and not self.mw.is_qa_mode:
                self.mw.is_qa_mode = True
                self.pause_timer_signal.emit()
                self.execute_student_qa_flow()
                self.mw.interrupt_requested = False
                self.mw.is_qa_mode = False
                self.resume_timer_signal.emit()
                if self.running:
                    self.update_status_signal.emit("Sapak dowam edýär...", "#4ade80")

            if not self.running:
                break

            try:
                sd.play(chunk, self.mw.sample_rate)
                sd.wait()
            except Exception as e:
                print("Audio playback notice:", e)

            self.mw.current_chunk_idx += 1

        if self.running:
            self.lesson_finished_signal.emit()

    def execute_student_qa_flow(self):
        try:
            self.update_status_signal.emit("Mugallym diňleýär: Soragyňyzy beriň (5 sekunt)...", "#ef4444")
            duration_sec = 5
            rec_sample_rate = 16000
            
            recording = sd.rec(int(duration_sec * rec_sample_rate), samplerate=rec_sample_rate, channels=1, dtype='int16')
            sd.wait()

            if not self.running:
                return

            self.update_status_signal.emit("Soragyňyz AI tarapyndan derňelýär...", "#38bdf8")

            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(rec_sample_rate)
                wf.writeframes(recording.tobytes())
            audio_bytes = buf.getvalue()

            client = genai.Client(api_key=self.mw.gemini_api_key)
            prompt_text = (
                f"Siz {self.mw.subject} mugallymy. Sapak mowzugy: {self.mw.topic}. "
                "Okuwçy sapak wagtynda goluny galdyryp şu soragy berdi. "
                "Haýyş, diňe Türkmen dilinde gysga, çalt we düşnükli jogap beriň (1-2 sözlem)."
            )
            
            response = client.models.generate_content(
                model='gemini-3.5-flash-lite',
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                    prompt_text
                ],
                config=types.GenerateContentConfig(
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    )
                )
            )
            answer_text = clean_latex_math(response.text)

            if not self.running:
                return

            self.update_status_signal.emit("Mugallym jogap berýär...", "#a855f7")
            ans_audio, ans_sr = synthesize_single_text_tts(answer_text, speed_factor=1.2)
            if ans_audio is not None and self.running:
                sd.play(ans_audio, ans_sr)
                sd.wait()

            if not self.running:
                return

            self.update_status_signal.emit("5 sekuntdan sapak dowam eder...", "#f59e0b")
            time.sleep(5)

        except Exception as e:
            print("Q&A Interrupt error:", e)
            if self.running:
                self.update_status_signal.emit("Ýalňyşlyk boldy, sapak dowam etdirilýär...", "#f87171")
                time.sleep(2)

    def stop(self):
        self.running = False
        sd.stop()
        self.wait()


# -------------------------------------------------------------------
# Main App Window
# -------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.exists(os.path.join(self.base_dir, "database")):
            self.base_dir = os.path.dirname(self.base_dir)

        self.db_path = os.path.join(self.base_dir, "database", "lessons.db")
        self.output_dir = os.path.join(self.base_dir, "output")

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        self.subject = "Matematika"
        self.topic = "Goşmak"
        self.duration_minutes = 10
        self.slide_pixmaps = []
        self.audio_chunks = []
        self.sample_rate = 16000
        self.current_slide_idx = 0
        self.current_chunk_idx = 0

        # Quiz mode variables
        self.quiz_questions = []
        self.current_quiz_idx = 0
        self.correct_count = 0
        self.incorrect_count = 0
        self.is_quiz_mode = False
        self.selected_answer_index = None

        self.interrupt_requested = False
        self.is_qa_mode = False

        self.slide_timer = QTimer(self)
        self.slide_timer.timeout.connect(self.auto_next_slide)

        self.playback_thread = None

        self.init_ui()
        self.showFullScreen()

        # Start Camera Thread
        self.camera_thread = CameraThread()
        self.camera_thread.hand_detected_signal.connect(self.on_hand_detected)
        self.camera_thread.gesture_detected_signal.connect(self.on_gesture_detected)
        self.camera_thread.start()

        self.load_stored_lesson_output()

    def closeEvent(self, event):
        if self.playback_thread:
            self.playback_thread.stop()
        if hasattr(self, 'camera_thread') and self.camera_thread:
            self.camera_thread.stop()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.slide_pixmaps and not self.is_quiz_mode:
            self.display_current_slide()

    def load_stored_lesson_output(self):
        json_path = os.path.join(self.output_dir, "lesson_data.json")
        if not os.path.exists(json_path):
            self.status_label.setText("Ýalňyşlyk: Sapak maglumatlary 'output' papkasynda tapylmady!")
            self.status_label.setStyleSheet("color: #f87171; background: transparent;")
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.subject = meta.get("subject", self.subject)
            self.topic = meta.get("topic", self.topic)
            self.duration_minutes = meta.get("duration_minutes", self.duration_minutes)
            self.sample_rate = meta.get("sample_rate", 16000)

            self.subject_label.setText(self.subject)
            self.topic_label.setText(f"Mowzuk: {self.topic}")

            # Load Slide Images
            self.slide_pixmaps = []
            for img_path in meta.get("slides", []):
                if os.path.exists(img_path):
                    self.slide_pixmaps.append(QPixmap(img_path))

            # Load Audio Chunks
            self.audio_chunks = []
            for wav_path in meta.get("audio_chunks", []):
                if os.path.exists(wav_path):
                    with wave.open(wav_path, 'rb') as wf:
                        frames = wf.readframes(wf.getnframes())
                        audio_arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                        self.audio_chunks.append(audio_arr)

            # Load Quiz Questions from Database
            self.load_quiz_from_db()

            self.set_status_ui("Sapak başlandy! Kamera taýýar (Gol galdyryp bilersiňiz).", "#4ade80")

            self.current_slide_idx = 0
            self.current_chunk_idx = 0

            self.display_current_slide()
            self.camera_thread.enabled = True

            # Start Audio Playback Thread
            self.playback_thread = PlaybackThread(self)
            self.playback_thread.update_status_signal.connect(self.set_status_ui)
            self.playback_thread.pause_timer_signal.connect(self.slide_timer.stop)
            self.playback_thread.resume_timer_signal.connect(self.display_current_slide)
            self.playback_thread.lesson_finished_signal.connect(self.start_gesture_quiz_mode)
            self.playback_thread.start()

        except Exception as e:
            self.status_label.setText(f"Maglumaty ýüklemekde ýalňyşlyk: {e}")
            self.status_label.setStyleSheet("color: #f87171; background: transparent;")

    def load_quiz_from_db(self):
        self.quiz_questions = []
        if not os.path.exists(self.db_path):
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT quiz_id FROM quizzes ORDER BY quiz_id DESC LIMIT 1")
            q_row = cursor.fetchone()
            if q_row:
                quiz_id = q_row[0]
                cursor.execute("SELECT question_text, answers_json, correct_answer FROM questions WHERE quiz_id = ?", (quiz_id,))
                for row in cursor.fetchall():
                    self.quiz_questions.append({
                        "question_text": row[0],
                        "answers": json.loads(row[1]),
                        "correct_answer": row[2]
                    })
            conn.close()
        except Exception as e:
            print("Error loading quiz from database:", e)

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

    def set_status_ui(self, message, color_hex="#38bdf8"):
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color_hex}; background: transparent;")

    def display_current_slide(self):
        if self.is_quiz_mode or not self.slide_pixmaps:
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

    def next_slide(self):
        if self.slide_pixmaps and self.current_slide_idx < len(self.slide_pixmaps) - 1:
            self.current_slide_idx += 1
            self.display_current_slide()

    def prev_slide(self):
        if self.slide_pixmaps and self.current_slide_idx > 0:
            self.current_slide_idx -= 1
            self.display_current_slide()

    def on_hand_detected(self):
        if not self.is_qa_mode and not self.interrupt_requested and not self.is_quiz_mode:
            self.interrupt_requested = True
            self.set_status_ui("Gol galdyryldy! Sözlem tamamlanansoň sapak duruzylýar...", "#f59e0b")

    def start_gesture_quiz_mode(self):
        self.slide_timer.stop()
        self.is_quiz_mode = True
        self.camera_thread.quiz_mode = True
        self.current_quiz_idx = 0
        self.correct_count = 0
        self.incorrect_count = 0

        if not self.quiz_questions:
            self.set_status_ui("Sapak tamamlandy! Test soraglary tapylmady.", "#a855f7")
            return

        self.set_status_ui("Sapak tamamlandy! Kamera Hereketleri arkaly Test başlandy.", "#38bdf8")
        self.display_quiz_question()

    def display_quiz_question(self):
        if self.current_quiz_idx >= len(self.quiz_questions):
            # Show final results
            total = self.correct_count + self.incorrect_count
            pct = (self.correct_count / total * 100) if total > 0 else 0
            res_text = f"Test tamamlandy! Dogry: {self.correct_count}, Nädogry: {self.incorrect_count} ({pct:.1f}%)"
            self.subject_label.setText("Test Netijesi")
            self.topic_label.setText(res_text)
            self.slide_display.setText(f"<h2 style='color: white;'>{res_text}</h2>")
            self.slide_counter.setText("Netije")
            return

        q = self.quiz_questions[self.current_quiz_idx]
        q_text = f"Sorag {self.current_quiz_idx + 1}/{len(self.quiz_questions)}: {q['question_text']}<br><br>"
        for i, ans in enumerate(q['answers']):
            q_text += f"<b>{i+1})</b> {ans}&nbsp;&nbsp;&nbsp;&nbsp;"
        
        self.slide_display.setText(f"<div style='color: #f8fafc; font-size: 20px; padding: 20px;'>{q_text}</div>")
        self.slide_counter.setText(f"Test Soragy {self.current_quiz_idx + 1} / {len(self.quiz_questions)}")
        self.set_status_ui("Sag eliňiz bilen 1-4 barmak görkezip jogap saýlaň. Çep el bilen kabul ediň ('accept').", "#38bdf8")

    def on_gesture_detected(self, right_gesture, left_gesture):
        if not self.is_quiz_mode or self.current_quiz_idx >= len(self.quiz_questions):
            return

        if right_gesture in ['1', '2', '3', '4']:
            self.selected_answer_index = int(right_gesture) - 1
            q = self.quiz_questions[self.current_quiz_idx]
            self.set_status_ui(f"Jogap saýlandy: {right_gesture}) {q['answers'][self.selected_answer_index]}. Çep eliňiz bilen tassyklaň.", "#f59e0b")

        if left_gesture == 'accept' and self.selected_answer_index is not None:
            q = self.quiz_questions[self.current_quiz_idx]
            if self.selected_answer_index == q['correct_answer']:
                self.correct_count += 1
                self.set_status_ui("Dogry jogap!", "#4ade80")
            else:
                self.incorrect_count += 1
                self.set_status_ui("Nädogry jogap!", "#f87171")
            
            self.selected_answer_index = None
            self.current_quiz_idx += 1
            QTimer.singleShot(1500, self.display_quiz_question)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())