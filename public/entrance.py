import sys
import os
import sqlite3
import subprocess
from datetime import datetime
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, 
    QGraphicsDropShadowEffect
)
from PyQt6.QtGui import QColor, QFont, QPalette, QLinearGradient, QBrush

class EntranceWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        # Base project directory
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

                # Parse date and time into datetime object
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

        # Background Gradient
        self.setAutoFillBackground(True)
        palette = self.palette()
        gradient = QLinearGradient(0, 0, 600, 750)
        gradient.setColorAt(0.0, QColor("#0f172a"))
        gradient.setColorAt(0.5, QColor("#1e1b4b"))
        gradient.setColorAt(1.0, QColor("#311042"))
        palette.setBrush(QPalette.ColorRole.Window, QBrush(gradient))
        self.setPalette(palette)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)

        # Glassmorphic Card Container
        card = QWidget(self)
        card.setObjectName("GlassCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(35, 45, 35, 45)
        card_layout.setSpacing(20)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(0, 0, 0, 120))
        shadow.setOffset(0, 10)
        card.setGraphicsEffect(shadow)

        # Subject Title (H1 Size)
        self.subject_label = QLabel(self.subject)
        self.subject_label.setFont(QFont("Segoe UI", 32, QFont.Weight.Bold))
        self.subject_label.setStyleSheet("color: #ffffff; background: transparent;")
        self.subject_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.subject_label)

        # Topic Subtitle
        if self.topic:
            self.topic_label = QLabel(f"Topic: {self.topic}")
            self.topic_label.setFont(QFont("Segoe UI", 13))
            self.topic_label.setStyleSheet("color: #a855f7; background: transparent; font-weight: 600;")
            self.topic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(self.topic_label)

        card_layout.addSpacing(15)

        # Countdown Section Heading
        countdown_heading = QLabel("TIME REMAINING TO START")
        countdown_heading.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        countdown_heading.setStyleSheet("color: #94a3b8; background: transparent; letter-spacing: 1px;")
        countdown_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(countdown_heading)

        # Countdown Display Box
        self.countdown_label = QLabel("00:00:00")
        self.countdown_label.setFont(QFont("Segoe UI", 36, QFont.Weight.Bold))
        self.countdown_label.setStyleSheet("""
            color: #38bdf8;
            background-color: rgba(0, 0, 0, 0.25);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 15px;
            padding: 15px;
        """)
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.countdown_label)

        card_layout.addSpacing(15)

        # Duration Display
        self.duration_label = QLabel(f"Lesson Duration: {self.duration_minutes} minutes")
        self.duration_label.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
        self.duration_label.setStyleSheet("color: #e2e8f0; background: transparent;")
        self.duration_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.duration_label)

        main_layout.addWidget(card)
        self.setLayout(main_layout)

        # Glassmorphism Styling
        self.setStyleSheet("""
            QWidget#GlassCard {
                background-color: rgba(255, 255, 255, 0.07);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 20px;
            }
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
        """Launches public/main.py and closes the entrance window."""
        main_script = os.path.join(self.base_dir, "public", "main.py")
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