import sys
import os
import sqlite3
import subprocess
from PyQt6.QtCore import Qt, QDate, QTime
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QGraphicsDropShadowEffect, QMessageBox
)
from PyQt6.QtGui import QColor, QFont, QPalette, QLinearGradient, QBrush

class ParentInputWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_db()
        self.init_ui()

    def init_db(self):
        """Creates the database directory and lessons table if they do not exist."""
        os.makedirs("database", exist_ok=True)
        self.db_path = os.path.join("database", "lessons.db")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                start_date TEXT NOT NULL,
                start_time TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def init_ui(self):
        self.setWindowTitle("AI Teacher Setup - Parent Dashboard")
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
        card_layout.setContentsMargins(35, 35, 35, 35)
        card_layout.setSpacing(16)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(0, 0, 0, 120))
        shadow.setOffset(0, 10)
        card.setGraphicsEffect(shadow)

        # Larger Header Title Block
        title = QLabel("Setup Lesson")
        title.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        title.setStyleSheet("color: #ffffff; background: transparent;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title)

        subtitle = QLabel("Configure your child's AI learning session")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setStyleSheet("color: #94a3b8; background: transparent;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(subtitle)

        card_layout.addSpacing(10)

        # Subject Input
        card_layout.addWidget(self.create_label("Subject"))
        self.subject_input = QLineEdit()
        self.subject_input.setPlaceholderText("e.g. Math, IT, Science")
        card_layout.addWidget(self.subject_input)

        # Topic Input
        card_layout.addWidget(self.create_label("Topic"))
        self.topic_input = QLineEdit()
        self.topic_input.setPlaceholderText("e.g. Calculus, Fractions, Web Dev")
        card_layout.addWidget(self.topic_input)

        # Date & Time Row (Standard Manual Input)
        dt_layout = QHBoxLayout()
        dt_layout.setSpacing(15)

        date_box = QVBoxLayout()
        date_box.addWidget(self.create_label("Start Date"))
        self.date_input = QLineEdit()
        self.date_input.setText(QDate.currentDate().toString("yyyy-MM-dd"))
        self.date_input.setPlaceholderText("YYYY-MM-DD")
        date_box.addWidget(self.date_input)

        time_box = QVBoxLayout()
        time_box.addWidget(self.create_label("Start Time"))
        self.time_input = QLineEdit()
        self.time_input.setText(QTime.currentTime().toString("HH:mm"))
        self.time_input.setPlaceholderText("HH:MM")
        time_box.addWidget(self.time_input)

        dt_layout.addLayout(date_box)
        dt_layout.addLayout(time_box)
        card_layout.addLayout(dt_layout)

        # Duration Input (Standard Manual Input)
        card_layout.addWidget(self.create_label("Duration (minutes)"))
        self.duration_input = QLineEdit()
        self.duration_input.setText("30")
        self.duration_input.setPlaceholderText("e.g. 30")
        card_layout.addWidget(self.duration_input)

        card_layout.addSpacing(15)

        # Submit Button
        self.submit_btn = QPushButton("Start AI Lesson")
        self.submit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.submit_btn.clicked.connect(self.save_to_db)
        card_layout.addWidget(self.submit_btn)

        main_layout.addWidget(card)
        self.setLayout(main_layout)

        # Glassmorphism Stylesheet
        self.setStyleSheet("""
            QWidget#GlassCard {
                background-color: rgba(255, 255, 255, 0.07);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 20px;
            }
            QLabel {
                color: #e2e8f0;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                font-weight: 600;
                background: transparent;
            }
            QLineEdit {
                background-color: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 10px;
                padding: 12px 14px;
                color: #ffffff;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #a855f7;
                background-color: rgba(255, 255, 255, 0.12);
            }
            QPushButton {
                background-color: #8b5cf6;
                color: #ffffff;
                font-family: 'Segoe UI', sans-serif;
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-radius: 12px;
                padding: 14px;
            }
            QPushButton:hover {
                background-color: #7c3aed;
            }
            QPushButton:pressed {
                background-color: #6d28d9;
            }
        """)

    def create_label(self, text):
        return QLabel(text)

    def save_to_db(self):
        subject = self.subject_input.text().strip()
        topic = self.topic_input.text().strip()
        start_date = self.date_input.text().strip()
        start_time = self.time_input.text().strip()
        duration = self.duration_input.text().strip()

        if not subject or not topic or not start_date or not start_time or not duration:
            QMessageBox.warning(self, "Input Error", "Please fill in all fields before starting.")
            return

        try:
            duration_val = int(duration)
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Duration must be a valid number.")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO lessons (subject, topic, start_date, start_time, duration_minutes)
            VALUES (?, ?, ?, ?, ?)
        """, (subject, topic, start_date, start_time, duration_val))
        conn.commit()
        conn.close()

        # Resolve paths accurately between admin/ and root directory
        current_dir = os.path.dirname(os.path.abspath(__file__)) # points to admin/
        root_dir = os.path.dirname(current_dir)                 # points to root directory
        generator_script = os.path.join(root_dir, "generator.py")

        if os.path.exists(generator_script):
            # Launch generator.py with its working directory set to the root folder
            subprocess.Popen([sys.executable, generator_script], cwd=root_dir)
        else:
            QMessageBox.warning(self, "Path Error", f"Could not find generator.py at: {generator_script}")

        # Close the GUI window immediately upon saving
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ParentInputWindow()
    window.show()
    sys.exit(app.exec())