import sys
import os
import sqlite3
import subprocess
from PyQt6.QtCore import Qt, QDate, QTime, QVariantAnimation, QEasingCurve, QRectF
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QGraphicsDropShadowEffect, QMessageBox,
)
from PyQt6.QtGui import QColor, QFont, QPainter, QPen


class AnimatedButton(QPushButton):
    """Paint-only animation: layout and normal button behavior stay intact."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(54)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        self._hover = 0.0
        self._press = 0.0
        self._hover_animation = QVariantAnimation(self)
        self._hover_animation.setDuration(180)
        self._hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._hover_animation.valueChanged.connect(self._set_hover)
        self._press_animation = QVariantAnimation(self)
        self._press_animation.setDuration(100)
        self._press_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._press_animation.valueChanged.connect(self._set_press)
        self.pressed.connect(lambda: self._animate_press(1.0))
        self.released.connect(lambda: self._animate_press(0.0))

    def _set_hover(self, value):
        self._hover = float(value)
        self.update()

    def _set_press(self, value):
        self._press = float(value)
        self.update()

    def _animate_press(self, target):
        self._press_animation.stop()
        self._press_animation.setStartValue(self._press)
        self._press_animation.setEndValue(target)
        self._press_animation.start()

    def _animate_hover(self, target):
        self._hover_animation.stop()
        self._hover_animation.setStartValue(self._hover)
        self._hover_animation.setEndValue(target)
        self._hover_animation.start()

    def enterEvent(self, event):
        self._animate_hover(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animate_hover(0.0)
        super().leaveEvent(event)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Animate the painted surface, never the widget's layout geometry.
        inset = self._press
        rect = QRectF(self.rect()).adjusted(3 + inset, 3 + inset, -3 - inset, -3 - inset)
        base, hover = QColor("#31594B"), QColor("#244739")
        color = QColor(*[
            round(a + (b - a) * self._hover)
            for a, b in zip(base.getRgb()[:3], hover.getRgb()[:3])
        ])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(rect, 9, 9)
        painter.setPen(QColor("#FFFFFF"))
        painter.setFont(self.font())
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())
        # A small arrow glides right on hover without shifting the label.
        x = rect.right() - 25 + 4 * self._hover
        y = rect.center().y()
        painter.setPen(QPen(QColor("#FFFFFF"), 1.6, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        from PyQt6.QtCore import QPointF
        painter.drawLine(QPointF(x - 8, y), QPointF(x + 2, y))
        painter.drawLine(QPointF(x - 2, y - 4), QPointF(x + 2, y))
        painter.drawLine(QPointF(x - 2, y + 4), QPointF(x + 2, y))
        if self.hasFocus():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#7D9E8E"), 1.5))
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.8, 0.8, -0.8, -0.8), 12, 12)


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
        self.setWindowTitle("Admin paneli")
        self.setFixedSize(600, 750)
        self.setObjectName("ParentWindow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(36, 30, 36, 30)
        main_layout.setSpacing(18)

        top_row = QHBoxLayout()
        brand = QLabel("Okuw")
        brand.setObjectName("Brand")
        top_row.addWidget(brand)
        top_row.addStretch()
        panel_label = QLabel("Admin paneli")
        panel_label.setObjectName("PanelLabel")
        top_row.addWidget(panel_label)
        main_layout.addLayout(top_row)

        card = QWidget(self)
        card.setObjectName("Card")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(32, 30, 32, 28)
        card_layout.setSpacing(0)

        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(24)
        shadow.setColor(QColor(32, 45, 38, 18))
        shadow.setOffset(0, 5)
        card.setGraphicsEffect(shadow)

        title = QLabel("Sapak başlat")
        title.setObjectName("Title")
        card_layout.addWidget(title)
        card_layout.addSpacing(8)
        subtitle = QLabel("Çagaňyzyň sapagyny sazlaň")
        subtitle.setObjectName("Subtitle")
        card_layout.addWidget(subtitle)
        card_layout.addSpacing(26)

        card_layout.addWidget(self.create_label("Sapak"))
        card_layout.addSpacing(8)
        self.subject_input = QLineEdit()
        self.subject_input.setPlaceholderText("Matematika, Informatika we ş.m")
        card_layout.addWidget(self.subject_input)
        card_layout.addSpacing(20)

        card_layout.addWidget(self.create_label("Tema"))
        card_layout.addSpacing(8)
        self.topic_input = QLineEdit()
        self.topic_input.setPlaceholderText("Funksiýalar, tor ulgamlary we ş.m")
        card_layout.addWidget(self.topic_input)
        card_layout.addSpacing(20)

        dt_layout = QHBoxLayout()
        dt_layout.setSpacing(16)
        date_box = QVBoxLayout()
        date_box.setSpacing(8)
        date_box.addWidget(self.create_label("Başlaýan senesi"))
        self.date_input = QLineEdit()
        self.date_input.setText(QDate.currentDate().toString("yyyy-MM-dd"))
        self.date_input.setPlaceholderText("YYYY-MM-DD")
        date_box.addWidget(self.date_input)
        time_box = QVBoxLayout()
        time_box.setSpacing(8)
        time_box.addWidget(self.create_label("Başlaýan wagty"))
        self.time_input = QLineEdit()
        self.time_input.setText(QTime.currentTime().toString("HH:mm"))
        self.time_input.setPlaceholderText("HH:MM")
        time_box.addWidget(self.time_input)
        dt_layout.addLayout(date_box)
        dt_layout.addLayout(time_box)
        card_layout.addLayout(dt_layout)
        card_layout.addSpacing(20)

        card_layout.addWidget(self.create_label("Dowamlylygy (minut)"))
        card_layout.addSpacing(8)
        self.duration_input = QLineEdit()
        self.duration_input.setText("15")
        self.duration_input.setPlaceholderText("meselem 15")
        card_layout.addWidget(self.duration_input)
        card_layout.addStretch(1)
        card_layout.addSpacing(24)
        self.submit_btn = AnimatedButton("Sapak döret")
        self.submit_btn.clicked.connect(self.save_to_db)
        card_layout.addWidget(self.submit_btn)
        main_layout.addWidget(card, 1)

        self.setStyleSheet("""
            QWidget#ParentWindow { background: #F5F5F1; }
            QWidget#Card {
                background: #FFFFFF; border: 1px solid #E3E6DF;
                border-radius: 16px;
            }
            QLabel {
                color: #35423B; background: transparent;
                font-family: 'Segoe UI'; font-size: 13px; font-weight: 600;
            }
            QLabel#Brand { color: #31594B; font-size: 19px; font-weight: 700; }
            QLabel#PanelLabel { color: #747E76; font-size: 12px; font-weight: 400; }
            QLabel#Title { color: #202D25; font-size: 29px; font-weight: 600; }
            QLabel#Subtitle { color: #778078; font-size: 14px; font-weight: 400; }
            QLineEdit {
                background: #FAFBF9; color: #26382E;
                border: 1px solid #DDE3DA; border-radius: 8px;
                padding: 12px; font-family: 'Segoe UI'; font-size: 14px;
                selection-background-color: #D7E7DD; selection-color: #203C2E;
            }
            QLineEdit:hover { border-color: #B4C3B6; }
            QLineEdit:focus { border-color: #527C65; background: #FFFFFF; }
            QMessageBox { background: #F5F5F1; }
            QMessageBox QLabel { color: #26382E; }
            QMessageBox QPushButton {
                background: #31594B; color: white; border: none;
                border-radius: 6px; padding: 8px 20px; min-width: 60px;
            }
            QMessageBox QPushButton:hover { background: #244739; }
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
            QMessageBox.warning(self, "Ýalňyşlyk", "Hemme öýjikleri doldyryň.")
            return

        try:
            duration_val = int(duration)
        except ValueError:
            QMessageBox.warning(self, "Ýaňyşlyk", "Dowamylyga san giriziň")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO lessons (subject, topic, start_date, start_time, duration_minutes)
            VALUES (?, ?, ?, ?, ?)
        """, (subject, topic, start_date, start_time, duration_val))
        conn.commit()
        conn.close()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        generator_script = os.path.join(root_dir, "generator.py")

        if os.path.exists(generator_script):
            subprocess.Popen([sys.executable, generator_script], cwd=root_dir)
        else:
            QMessageBox.warning(self, "Path Error", f"Could not find generator.py at: {generator_script}")

        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ParentInputWindow()
    window.show()
    sys.exit(app.exec())
