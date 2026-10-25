import sys
import os
import sqlite3
import subprocess
from datetime import datetime
from PyQt6.QtCore import Qt, QTimer, QVariantAnimation, QEasingCurve, QRectF
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGraphicsDropShadowEffect,
)
from PyQt6.QtGui import QColor, QPainter, QPen


class CountdownLabel(QLabel):
    """A quiet border pulse on each tick; the digits remain fully readable."""

    def __init__(self, parent=None):
        super().__init__("00:00:00", parent)
        self._pulse = 0.0
        self.setObjectName("Countdown")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(116)
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(650)
        self._animation.setStartValue(1.0)
        self._animation.setEndValue(0.0)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.valueChanged.connect(self._set_pulse)

    def _set_pulse(self, value):
        self._pulse = float(value)
        self.update()

    def setText(self, text):
        changed = text != self.text()
        super().setText(text)
        if changed and hasattr(self, "_animation"):
            self._animation.stop()
            if ":" in text:
                self._animation.start()
            else:
                self._set_pulse(0.0)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        resting, active = QColor("#DDE6DC"), QColor("#91B09A")
        border = QColor(*[
            round(a + (b - a) * self._pulse)
            for a, b in zip(resting.getRgb()[:3], active.getRgb()[:3])
        ])
        painter.setPen(QPen(border, 1.2))
        painter.setBrush(QColor("#F3F7F1"))
        painter.drawRoundedRect(
            QRectF(self.rect()).adjusted(1, 1, -1, -1), 12, 12
        )
        painter.end()
        super().paintEvent(event)


class EntranceWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Base project directory: retain the original folder convention.
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(self.base_dir, "database", "lessons.db")

        self.subject = "No Lesson Scheduled"
        self.topic = ""
        self.target_datetime = None
        self.duration_minutes = 0

        self.load_latest_lesson()
        self.init_ui()
        self.start_countdown_timer()

    def load_latest_lesson(self):
        """Fetches the most recent lesson record from database/lessons.db."""
        if not os.path.exists(self.db_path):
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT subject, topic, start_date, start_time, duration_minutes
                FROM lessons
                ORDER BY id DESC LIMIT 1
            """)
            row = cursor.fetchone()
            conn.close()

            if row:
                self.subject = row[0]
                self.topic = row[1]
                date_str = row[2]
                time_str = row[3]
                self.duration_minutes = row[4]

                dt_str = f"{date_str} {time_str}"
                try:
                    self.target_datetime = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
                except ValueError:
                    self.target_datetime = datetime.now()
        except sqlite3.Error as e:
            print("Database connection error:", e)

    def init_ui(self):
        self.setWindowTitle("AI Teacher - Lesson Entrance")
        self.setFixedSize(600, 750)
        self.setObjectName("EntranceWindow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(36, 30, 36, 30)
        main_layout.setSpacing(18)

        top_row = QHBoxLayout()
        brand = QLabel("Okuw")
        brand.setObjectName("Brand")
        top_row.addWidget(brand)
        top_row.addStretch()
        section = QLabel("Lesson entrance")
        section.setObjectName("Section")
        top_row.addWidget(section)
        main_layout.addLayout(top_row)

        card = QWidget(self)
        card.setObjectName("Card")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(32, 36, 32, 32)
        card_layout.setSpacing(0)

        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(24)
        shadow.setColor(QColor(32, 45, 38, 18))
        shadow.setOffset(0, 5)
        card.setGraphicsEffect(shadow)

        eyebrow = QLabel("YOUR LESSON")
        eyebrow.setObjectName("Eyebrow")
        card_layout.addWidget(eyebrow)
        card_layout.addSpacing(14)

        self.subject_label = QLabel(self.subject)
        self.subject_label.setObjectName("Subject")
        self.subject_label.setTextFormat(Qt.TextFormat.PlainText)
        self.subject_label.setWordWrap(True)
        card_layout.addWidget(self.subject_label)

        if self.topic:
            card_layout.addSpacing(12)
            self.topic_label = QLabel(f"Topic: {self.topic}")
            self.topic_label.setObjectName("Topic")
            self.topic_label.setTextFormat(Qt.TextFormat.PlainText)
            self.topic_label.setWordWrap(True)
            card_layout.addWidget(self.topic_label)

        card_layout.addStretch(1)
        card_layout.addSpacing(28)

        countdown_heading = QLabel("TIME REMAINING TO START")
        countdown_heading.setObjectName("CountdownHeading")
        countdown_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(countdown_heading)
        card_layout.addSpacing(14)

        self.countdown_label = CountdownLabel()
        card_layout.addWidget(self.countdown_label)
        card_layout.addSpacing(20)

        self.duration_label = QLabel(
            f"Lesson Duration: {self.duration_minutes} minutes"
        )
        self.duration_label.setObjectName("Duration")
        self.duration_label.setWordWrap(True)
        self.duration_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.duration_label)
        card_layout.addStretch(1)

        main_layout.addWidget(card, 1)

        self.setStyleSheet("""
            QWidget#EntranceWindow { background: #F5F5F1; }
            QWidget#Card {
                background: #FFFFFF;
                border: 1px solid #E3E6DF;
                border-radius: 16px;
            }
            QLabel {
                background: transparent; color: #35423B;
                font-family: 'Segoe UI'; font-size: 14px;
            }
            QLabel#Brand {
                color: #31594B; font-size: 19px; font-weight: 700;
            }
            QLabel#Section { color: #747E76; font-size: 12px; }
            QLabel#Eyebrow {
                color: #748176; font-size: 11px; font-weight: 600;
            }
            QLabel#Subject {
                color: #202D25; font-size: 32px; font-weight: 600;
            }
            QLabel#Topic { color: #728077; font-size: 15px; }
            QLabel#CountdownHeading {
                color: #778078; font-size: 11px; font-weight: 600;
            }
            QLabel#Countdown {
                color: #31594B; font-family: 'Consolas', 'DejaVu Sans Mono';
                font-size: 42px; font-weight: 600;
                background: transparent; border: none;
            }
            QLabel#Duration { color: #728077; font-size: 13px; }
        """)

    def start_countdown_timer(self):
        """Starts a timer that updates every second."""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_countdown)
        self.timer.start(1000)
        self.update_countdown()

    def update_countdown(self):
        """Updates remaining time and launches public/main.py when timer hits zero."""
        if not self.target_datetime:
            self.countdown_label.setText("No Schedule")
            return

        now = datetime.now()
        diff = self.target_datetime - now

        if diff.total_seconds() <= 0:
            self.timer.stop()
            self.launch_main_app()
        else:
            total_seconds = int(diff.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.countdown_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    def launch_main_app(self):
        """Launches main.py and closes the entrance window."""
        main_script = os.path.join(self.base_dir, "public/main.py")
        if os.path.exists(main_script):
            subprocess.Popen([sys.executable, main_script])
        else:
            print(f"File not found: {main_script}")

        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EntranceWindow()
    window.show()
    sys.exit(app.exec())
