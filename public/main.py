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

import torch
from transformers import VitsModel, AutoTokenizer
import cv2
import mediapipe as mp
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QSize, QTimer, QObject, QEvent,
    QVariantAnimation, QEasingCurve,
)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea,
)
from PyQt6.QtGui import QColor, QFont, QPalette, QBrush, QPixmap, QImage
from google import genai
from google.genai import types
import sounddevice as sd

LOCAL_TTS_MODEL = None
LOCAL_TTS_TOKENIZER = None


def get_local_tts():
    """Loads and caches the local Hugging Face TTS model for live responses."""
    global LOCAL_TTS_MODEL, LOCAL_TTS_TOKENIZER
    if LOCAL_TTS_MODEL is None or LOCAL_TTS_TOKENIZER is None:
        model_name = "facebook/mms-tts-tuk-script_latin"
        LOCAL_TTS_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        LOCAL_TTS_MODEL = VitsModel.from_pretrained(model_name)
    return LOCAL_TTS_MODEL, LOCAL_TTS_TOKENIZER


def clean_latex_math(text):
    """Converts raw LaTeX math syntax into Unicode for PyQt rendering."""
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
        'n': 'ⁿ', 'i': 'ⁱ', 'x': 'ˣ', 'k': 'ᵏ',
    }

    def sup_replace_brace(match):
        return "".join(sup_map.get(c, c) for c in match.group(1))

    def sup_replace_single(match):
        char = match.group(1)
        return sup_map.get(char, f"^{char}")

    text = re.sub(r'\^\{([^}]+)\}', sup_replace_brace, text)
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


class CameraThread(QThread):
    frame_signal = pyqtSignal(QImage)
    camera_status_signal = pyqtSignal(str)
    hand_detected_signal = pyqtSignal()
    gesture_detected_signal = pyqtSignal(str, str)
    face_detected_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.running = True
        self.enabled = False
        self.quiz_mode = False

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.camera_status_signal.emit("Kamera açylmady. Düwmeler arkaly jogap beriň.")
            return
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.7, min_tracking_confidence=0.5,
        )

        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)

        last_trigger = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.camera_status_signal.emit("Kamera bilen aragatnaşyk kesildi.")
                break
            if self.enabled:
                frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                h, w, _ = frame_rgb.shape

                # Face Detection Check
                face_results = face_detection.process(frame_rgb)
                is_face_present = bool(face_results.detections)
                self.face_detected_signal.emit(is_face_present)

                if self.quiz_mode:
                    self.frame_signal.emit(QImage(
                        frame_rgb.data, w, h, frame_rgb.strides[0],
                        QImage.Format.Format_RGB888,
                    ).copy())
                
                results = hands.process(frame_rgb)
                if not self.quiz_mode:
                    if results.multi_hand_landmarks:
                        now = time.time()
                        if now - last_trigger > 6:
                            last_trigger = now
                            self.hand_detected_signal.emit()
                else:
                    right_gesture = "None"
                    left_gesture = "None"
                    if results.multi_hand_landmarks and results.multi_handedness:
                        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                            handedness = results.multi_handedness[i].classification[0].label
                            landmarks = hand_landmarks.landmark

                            def is_up(tip_idx, pip_idx):
                                return landmarks[tip_idx].y < landmarks[pip_idx].y - 0.02

                            fingers = [is_up(8, 6), is_up(12, 10), is_up(16, 14), is_up(20, 18)]
                            num_up = sum(fingers)
                            is_thumb_up = landmarks[4].y < landmarks[3].y - 0.03
                            if handedness == "Right":
                                if num_up == 1 and fingers[0]:
                                    right_gesture = '1'
                                elif num_up == 2 and fingers[0] and fingers[1]:
                                    right_gesture = '2'
                                elif num_up == 3 and fingers[0] and fingers[1] and fingers[2]:
                                    right_gesture = '3'
                                elif num_up == 4 and all(fingers):
                                    right_gesture = '4'
                            elif handedness == "Left":
                                if is_thumb_up and num_up == 0:
                                    left_gesture = 'accept'
                                elif num_up == 4 and all(fingers):
                                    left_gesture = 'cancel'
                    self.gesture_detected_signal.emit(right_gesture, left_gesture)
            self.msleep(100)
        hands.close()
        face_detection.close()
        cap.release()

    def stop(self):
        self.running = False
        self.wait()


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
            # Check if face is present; if not, wait loop
            while self.running and not self.mw.face_detected:
                self.update_status_signal.emit("Ýüz ýüze tapylmady! Sapak garaşýar...", "#ef4444")
                time.sleep(0.5)

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
            self.update_status_signal.emit("Mugallym diňleýär: Soragyňyzy beriň...", "#ef4444")
            duration_sec = 5
            rec_sample_rate = 16000
            recording = sd.rec(
                int(duration_sec * rec_sample_rate), samplerate=rec_sample_rate,
                channels=1, dtype='int16',
            )
            sd.wait()
            if not self.running:
                return
            self.update_status_signal.emit("Soragyňyz mugalym tarapyndan pikirlenilýär...", "#38bdf8")
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
                "Ilki bilen berlen soragyň şu sapak mowzugyna ({self.mw.topic}) gönüden-göni degişli ýa-da dälledigini barlaň. "
                "Eger sorag şu sapağa ýa-da mowzugyna degişli däl bolsa, hökman diňe şu sözleri aýdyň: 'Seniň soragyň tema degişli däl'. "
                "Eger tema degişli bolsa, diňe Türkmen dilinde gysga, çalt we düşnükli jogap beriň (1-2 sözlem)."
            )
            response = client.models.generate_content(
                model='gemini-3.5-flash-lite',
                contents=[types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"), prompt_text],
                config=types.GenerateContentConfig(
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                ),
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
            self.update_status_signal.emit("Sapak dowam etdirilýär...", "#f59e0b")
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


STATUS_COLORS = {
    "#4ade80": "#31594B",
    "#38bdf8": "#286482",
    "#ef4444": "#AE3935",
    "#f87171": "#AE3935",
    "#a855f7": "#66527C",
    "#c084fc": "#66527C",
    "#d8b4fe": "#66527C",
    "#f59e0b": "#8A5A13",
}


def _mix(first, second, progress):
    a, b = QColor(first), QColor(second)
    return QColor(*[
        round(x + (y - x) * progress)
        for x, y in zip(a.getRgb()[:3], b.getRgb()[:3])
    ]).name()


class _ButtonStyle(QObject):
    def __init__(self, button, role="secondary"):
        super().__init__(button)
        self.button = button
        self.role = role
        self.selected = False
        self.progress = 0.0
        self.animation = QVariantAnimation(self)
        self.animation.setDuration(170)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.animation.valueChanged.connect(self._paint)
        button.installEventFilter(self)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.pressed.connect(lambda: self.animate(1.0))
        button.released.connect(lambda: self.animate(0.65 if button.underMouse() else 0.0))
        self.render()

    def animate(self, target):
        self.animation.stop()
        self.animation.setStartValue(self.progress)
        self.animation.setEndValue(target)
        self.animation.start()

    def _paint(self, value):
        self.progress = float(value)
        self.render()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Enter and self.button.isEnabled():
            self.animate(0.65)
        elif event.type() == QEvent.Type.Leave:
            self.animate(0.0)
        elif event.type() == QEvent.Type.EnabledChange:
            self.animation.stop()
            self.progress = 0.0
            self.render()
        return False

    def render(self):
        if self.role == "primary":
            base, hover, text, border = "#31594B", "#244739", "#FFFFFF", "#31594B"
        elif self.role == "exit":
            base, hover, text, border = "#FFFFFF", "#FAEAE7", "#99463E", "#E4D5D0"
        elif self.selected:
            base, hover, text, border = "#E8F1E9", "#D8E8DC", "#254A36", "#527C65"
        else:
            base, hover, text, border = "#FFFFFF", "#EEF3EC", "#35463B", "#DDE3DA"
        if not self.button.isEnabled():
            base = "#E8F1E9" if self.selected else "#F2F3F0"
            hover = base
            text = "#48604E" if self.selected else "#8A928B"
            border = "#527C65" if self.selected else "#E2E5DF"
        background = _mix(base, hover, self.progress)
        padding = "14px" if self.role == "answer" else "9px 14px"
        self.button.setStyleSheet(f"""
            QPushButton {{
                background: {background}; color: {text};
                border: 1px solid {border}; border-radius: 9px;
                padding: {padding}; font-family: 'Segoe UI';
                font-size: 13px; font-weight: 600;
            }}
            QPushButton:focus {{ border: 1px solid #527C65; }}
        """)
        if hasattr(self.button, "answer_label"):
            self.button.answer_label.setStyleSheet(
                f"color: {text}; background: transparent; border: none; "
                "font-family: 'Segoe UI'; font-size: 18px; font-weight: 500;"
            )


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
        self.quiz_questions = []
        self.current_quiz_idx = 0
        self.correct_count = 0
        self.incorrect_count = 0
        self.is_quiz_mode = False
        self.selected_answer_index = None
        self.answer_pending = False
        self.last_left_gesture = "None"
        self.last_right_gesture = "None"
        self.interrupt_requested = False
        self.is_qa_mode = False
        self.face_detected = False
        self.initial_face_wait_passed = False

        self.slide_timer = QTimer(self)
        self.slide_timer.timeout.connect(self.auto_next_slide)
        self.playback_thread = None
        self.init_ui()
        self.showFullScreen()
        
        self.camera_thread = CameraThread()
        self.camera_thread.hand_detected_signal.connect(self.on_hand_detected)
        self.camera_thread.gesture_detected_signal.connect(self.on_gesture_detected)
        self.camera_thread.frame_signal.connect(self.show_camera_frame)
        self.camera_thread.camera_status_signal.connect(self.camera_label.setText)
        self.camera_thread.face_detected_signal.connect(self.on_face_detected)
        self.camera_thread.start()
        
        # Enable camera immediately for startup face check
        self.camera_thread.enabled = True
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

    def on_face_detected(self, detected):
        self.face_detected = detected
        if not self.initial_face_wait_passed and detected:
            self.initial_face_wait_passed = True
            self.set_status_ui("Sapak başlandy! Düşünmedik ýagdaýyňyzda eliňizi galdyryp bilersiňiz.", "#4ade80")

    def load_stored_lesson_output(self):
        json_path = os.path.join(self.output_dir, "lesson_data.json")
        if not os.path.exists(json_path):
            self.set_status_ui("Ýalňyşlyk: Sapak maglumatlary 'output' papkasynda tapylmady!", "#f87171")
            return
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.subject = meta.get("subject", self.subject)
            self.topic = meta.get("topic", self.topic)
            self.duration_minutes = meta.get("duration_minutes", self.duration_minutes)
            self.sample_rate = meta.get("sample_rate", 16000)
            self.subject_label.setText(self.subject)
            self.slide_pixmaps = []
            for img_path in meta.get("slides", []):
                if os.path.exists(img_path):
                    self.slide_pixmaps.append(QPixmap(img_path))
            self.audio_chunks = []
            for wav_path in meta.get("audio_chunks", []):
                if os.path.exists(wav_path):
                    with wave.open(wav_path, 'rb') as wf:
                        frames = wf.readframes(wf.getnframes())
                        audio_arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                        self.audio_chunks.append(audio_arr)
            self.load_quiz_from_db()
            
            # Wait for face detection before confirming start
            self.set_status_ui("Okuwçynyň ýüzi garaşylýar... Kameranyň öňüne geçiň.", "#ef4444")
            
            self.current_slide_idx = 0
            self.current_chunk_idx = 0
            self.display_current_slide()
            
            self.playback_thread = PlaybackThread(self)
            self.playback_thread.update_status_signal.connect(self.set_status_ui)
            self.playback_thread.pause_timer_signal.connect(self.slide_timer.stop)
            self.playback_thread.resume_timer_signal.connect(self.display_current_slide)
            self.playback_thread.lesson_finished_signal.connect(self.start_gesture_quiz_mode)
            self.playback_thread.start()
        except Exception as e:
            self.set_status_ui(f"Maglumaty ýüklemekde ýalňyşlyk: {e}", "#f87171")

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
                        "question_text": row[0], "answers": json.loads(row[1]),
                        "correct_answer": row[2],
                    })
            conn.close()
        except Exception as e:
            print("Test maglumatlary ýüklemekde ýalňyşlyk:", e)

    def init_ui(self):
        self.setWindowTitle("Mugallym - Sapak")
        self.setObjectName("LessonWindow")
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setBrush(QPalette.ColorRole.Window, QBrush(QColor("#F5F5F1")))
        self.setPalette(palette)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 14, 18, 14)
        card = QWidget(self)
        card.setObjectName("LessonCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 14, 20, 14)
        card_layout.setSpacing(8)
        self.subject_label = QLabel(self.subject)
        self.subject_label.setObjectName("Subject")
        self.subject_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.subject_label)
        self.topic_label = QLabel(f"Mowzuk: {self.topic}")
        self.topic_label.setObjectName("Topic")
        self.topic_label.setWordWrap(True)
        self.topic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.topic_label)
        self.status_label = QLabel("Sapak ýüklenýär...")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_status_ui("Sapak ýüklenýär...")
        card_layout.addWidget(self.status_label)
        self.slide_display = QLabel()
        self.slide_display.setObjectName("SlideDisplay")
        self.slide_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.slide_display, stretch=1)
        self.build_quiz_panel(card_layout)
        nav_layout = QHBoxLayout()
        self.prev_btn = self.make_button("◀ Yza")
        self.prev_btn.setFixedWidth(120)
        self.prev_btn.clicked.connect(self.prev_slide)
        self.slide_counter = QLabel("Slaýd 0 / 0")
        self.slide_counter.setObjectName("SlideCounter")
        self.slide_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.next_btn = self.make_button("Öňe ▶")
        self.next_btn.setFixedWidth(120)
        self.next_btn.clicked.connect(self.next_slide)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.slide_counter)
        nav_layout.addWidget(self.next_btn)
        card_layout.addLayout(nav_layout)
        exit_btn = self.make_button("Çykmak (Esc)", "exit")
        exit_btn.setFixedWidth(140)
        exit_btn.clicked.connect(self.close)
        card_layout.addWidget(exit_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(card)
        self.setStyleSheet("""
            QWidget#LessonWindow { background: #F5F5F1; }
            QWidget#LessonCard {
                background: #FFFFFF; border: 1px solid #E3E6DF; border-radius: 16px;
            }
            QLabel { font-family: 'Segoe UI'; color: #35423B; background: transparent; }
            QLabel#Subject { color: #202D25; font-size: 28px; font-weight: 600; }
            QLabel#Topic { color: #728077; font-size: 14px; }
            QLabel#SlideDisplay {
                background: #F8F9F6; border: 1px solid #E4E8E0; border-radius: 10px;
            }
            QLabel#SlideCounter { color: #778078; font-size: 13px; }
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical { background: #F4F5F1; width: 8px; margin: 0; }
            QScrollBar::handle:vertical {
                background: #C7D3C6; border-radius: 4px; min-height: 30px;
            }
            QScrollBar::handle:vertical:hover { background: #9FB49F; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        """)

    def make_button(self, text, role="secondary"):
        button = QPushButton(text)
        button._light_style = _ButtonStyle(button, role)
        return button

    def btn_style(self, color_hex):
        color = STATUS_COLORS.get(color_hex.lower(), color_hex)
        return f"""
            QPushButton {{ background: #FFFFFF; color: {color}; border: 1px solid #DDE3DA;
                border-radius: 9px; padding: 9px 14px; font-size: 13px; font-weight: 600; }}
            QPushButton:hover {{ background: #EEF3EC; }}
            QPushButton:disabled {{ background: #F2F3F0; color: #8A928B; }}
        """

    def set_status_ui(self, message, color_hex="#38bdf8"):
        self.status_label.setText(message)
        color = STATUS_COLORS.get(color_hex.lower(), color_hex)
        self.status_label.setStyleSheet(
            f"color: {color}; background: transparent; font-size: 13px; font-weight: 500;"
        )

    def display_current_slide(self):
        if self.is_quiz_mode or not self.slide_pixmaps:
            return
        pixmap = self.slide_pixmaps[self.current_slide_idx]
        target_size = self.slide_display.contentsRect().size()
        if target_size.width() < 100 or target_size.height() < 100:
            target_size = QSize(960, 540)

        scaled = pixmap.scaled(
            target_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.slide_display.setPixmap(scaled)
        self.slide_counter.setText(f"Slaýd {self.current_slide_idx + 1} / {len(self.slide_pixmaps)}")
        slide_count = max(1, len(self.slide_pixmaps))
        slide_duration_ms = int((self.duration_minutes * 60 * 1000) / slide_count)
        self.slide_timer.start(slide_duration_ms)

    def auto_next_slide(self):
        if self.face_detected and self.slide_pixmaps and self.current_slide_idx < len(self.slide_pixmaps) - 1:
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
            self.set_status_ui("El galdyryldy! Sözlem tamamlanansoň sapak duruzylýar...", "#f59e0b")

    def build_quiz_panel(self, layout):
        self.quiz_panel = QWidget()
        self.quiz_panel.setStyleSheet("background: transparent;")
        row = QHBoxLayout(self.quiz_panel)
        row.setSpacing(20)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.viewport().setStyleSheet("background: transparent;")
        questions = QWidget()
        questions.setStyleSheet("background: transparent;")
        left = QVBoxLayout(questions)
        left.setSpacing(12)
        self.question_label = QLabel()
        self.question_label.setTextFormat(Qt.TextFormat.PlainText)
        self.question_label.setWordWrap(True)
        self.question_label.setStyleSheet(
            "color: #202D25; font-size: 24px; font-weight: 600; padding: 14px 8px;"
        )
        left.addWidget(self.question_label)
        self.answer_buttons = []
        for i in range(4):
            button = QPushButton()
            button.setMinimumHeight(76)
            label = QLabel(button)
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            inner = QVBoxLayout(button)
            inner.addWidget(label)
            button.answer_label = label
            button._light_style = _ButtonStyle(button, "answer")
            button.clicked.connect(lambda checked=False, index=i: self.select_answer(index))
            left.addWidget(button)
            self.answer_buttons.append(button)
        controls = QHBoxLayout()
        self.cancel_button = self.make_button("Saýlawy ýatyr")
        self.confirm_button = self.make_button("Jogaby tassykla", "primary")
        for button in (self.cancel_button, self.confirm_button):
            controls.addWidget(button)
        self.cancel_button.clicked.connect(self.clear_answer)
        self.confirm_button.clicked.connect(self.confirm_answer)
        left.addLayout(controls)
        left.addStretch()
        scroll.setWidget(questions)
        row.addWidget(scroll, 1)
        right = QVBoxLayout()
        self.camera_label = QLabel("Kamera garaşylýar...")
        self.camera_label.setMinimumSize(240, 180)
        self.camera_label.setWordWrap(True)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet(
            "color: #68796C; background: #F0F4EE; border: 1px solid #DDE6DA; "
            "border-radius: 12px; font-size: 14px;"
        )
        right.addWidget(self.camera_label, 1)
        self.gesture_label = QLabel("Sag el: 1–4 • Çep el: başam barmak — tassykla, açyk el — ýatyr")
        self.gesture_label.setWordWrap(True)
        self.gesture_label.setStyleSheet("color: #728077; font-size: 14px; padding: 12px 4px;")
        right.addWidget(self.gesture_label)
        row.addLayout(right, 1)
        layout.addWidget(self.quiz_panel, 1)
        self.quiz_panel.hide()

    def show_camera_frame(self, frame):
        if self.is_quiz_mode:
            self.camera_label.setPixmap(QPixmap.fromImage(frame).scaled(
                self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

    def start_gesture_quiz_mode(self):
        self.slide_timer.stop()
        self.is_quiz_mode = True
        self.camera_thread.quiz_mode = True
        self.current_quiz_idx = 0
        self.correct_count = self.incorrect_count = 0
        self.slide_display.hide()
        self.prev_btn.hide()
        self.next_btn.hide()
        self.quiz_panel.show()
        self.display_quiz_question()

    def refresh_answers(self):
        for i, button in enumerate(self.answer_buttons):
            button._light_style.selected = i == self.selected_answer_index
            button.setEnabled(not self.answer_pending)
            button._light_style.render()
        self.confirm_button.setEnabled(self.selected_answer_index is not None and not self.answer_pending)
        self.cancel_button.setEnabled(self.selected_answer_index is not None and not self.answer_pending)

    def display_quiz_question(self):
        self.answer_pending = False
        self.selected_answer_index = None
        if self.current_quiz_idx >= len(self.quiz_questions):
            self.answer_pending = True
            total = self.correct_count + self.incorrect_count
            pct = self.correct_count / total * 100 if total else 0
            self.question_label.setText(
                f"Test tamamlandy!\nDogry: {self.correct_count}\nNädogry: {self.incorrect_count}\nNetije: {pct:.1f}%"
                if total else "Test soraglary tapylmady."
            )
            for button in self.answer_buttons:
                button.hide()
            self.confirm_button.hide()
            self.cancel_button.hide()
            self.slide_counter.setText("Netije")
            self.set_status_ui("Sapak tamamlandy.", "#c084fc")
            return
        q = self.quiz_questions[self.current_quiz_idx]
        self.question_label.setText(q['question_text'])
        for i, button in enumerate(self.answer_buttons):
            button.answer_label.setText(f"{i+1})  {q['answers'][i]}")
        self.refresh_answers()
        self.slide_counter.setText(f"Sorag {self.current_quiz_idx + 1} / {len(self.quiz_questions)}")
        self.set_status_ui("Jogap saýlaň we tassyklaň.", "#c084fc")

    def select_answer(self, index):
        if self.answer_pending or not self.is_quiz_mode:
            return
        self.selected_answer_index = index
        self.refresh_answers()
        self.set_status_ui(f"{index + 1}-nji jogap saýlandy. Tassyklap ýa-da üýtgedip bilersiňiz.", "#d8b4fe")

    def clear_answer(self):
        if not self.answer_pending:
            self.selected_answer_index = None
            self.refresh_answers()
            self.set_status_ui("Saýlaw ýatyryldy.", "#c084fc")

    def confirm_answer(self):
        if self.answer_pending or self.selected_answer_index is None:
            return
        self.answer_pending = True
        q = self.quiz_questions[self.current_quiz_idx]
        correct = self.selected_answer_index == q['correct_answer']
        self.correct_count += int(correct)
        self.incorrect_count += int(not correct)
        self.refresh_answers()
        self.set_status_ui("Jogabyňyz kabul edildi.", "#c084fc")
        self.current_quiz_idx += 1
        QTimer.singleShot(1500, self.display_quiz_question)

    def on_gesture_detected(self, right_gesture, left_gesture):
        new_right = right_gesture != self.last_right_gesture
        new_left = left_gesture != self.last_left_gesture
        self.last_right_gesture, self.last_left_gesture = right_gesture, left_gesture
        if not self.is_quiz_mode or self.answer_pending:
            return
        if new_right and right_gesture in ('1', '2', '3', '4'):
            self.select_answer(int(right_gesture) - 1)
        if new_left and left_gesture == 'accept':
            self.confirm_answer()
        elif new_left and left_gesture == 'cancel':
            self.clear_answer()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())