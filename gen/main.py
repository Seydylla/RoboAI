# import sys
# import os
# import sqlite3
# import warnings
# import io
# import re
# import wave
# import numpy as np

# warnings.filterwarnings("ignore")

# from dotenv import load_dotenv
# load_dotenv()

# import torch
# from transformers import VitsModel, AutoTokenizer

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# from PyQt6.QtCore import Qt, QRect
# from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QBrush
# from PyQt6.QtWidgets import QApplication

# from google import genai
# from google.genai import types

# # Load Hugging Face TTS Model
# model_name = "facebook/mms-tts-tuk-script_latin"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = VitsModel.from_pretrained(model_name)


# def clean_latex_math(text):
#     if not text:
#         return ""
#     replacements = {
#         r'\cdot': '·', r'\times': '×', r'\div': '÷', r'\pm': '±',
#         r'\infty': '∞', r'\pi': 'π', r'\alpha': 'α', r'\beta': 'β',
#         r'\theta': 'θ', r'\le': '≤', r'\leq': '≤', r'\ge': '≥',
#         r'\geq': '≥', r'\neq': '≠', r'\approx': '≈', r'\sqrt': '√',
#         r'\int': '∫', r'\sum': '∑',
#     }
#     for k, v in replacements.items():
#         text = text.replace(k, v)
        
#     sup_map = {
#         '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
#         '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
#         '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
#         'n': 'ⁿ', 'i': 'ⁱ', 'x': 'ˣ', 'k': 'ᵏ'
#     }
    
#     def sup_replace_brace(match):
#         return "".join(sup_map.get(c, c) for c in match.group(1))
#     text = re.sub(r'\^\{([^}]+)\}', sup_replace_brace, text)
    
#     def sup_replace_single(match):
#         return sup_map.get(match.group(1), f"^{match.group(1)}")
#     text = re.sub(r'\^([0-9nixk+\-])', sup_replace_single, text)
    
#     text = text.replace('$', '')
#     return re.sub(r' +', ' ', text)


# def generate_topic_graph(slide_index, subject, topic, slide_title=""):
#     fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=100)
#     fig.patch.set_facecolor('#1e293b')
#     ax.set_facecolor('#0f172a')

#     ax.spines['bottom'].set_color('#334155')
#     ax.spines['top'].set_color('#334155')
#     ax.spines['right'].set_color('#334155')
#     ax.spines['left'].set_color('#334155')
#     ax.tick_params(axis='x', colors='#94a3b8')
#     ax.tick_params(axis='y', colors='#94a3b8')
#     ax.title.set_color('#f8fafc')

#     full_topic = f"{topic} {subject} {slide_title}".lower()

#     if any(w in full_topic for w in ["goşmak", "aýyrmak", "kópleltmek", "bölmek", "addition", "subtraction"]):
#         if slide_index % 2 == 0:
#             ax.axhline(0, color='#94a3b8', linewidth=2)
#             ax.plot([0, 2], [0, 0.5], color='#38bdf8', linewidth=3, label='+2')
#             ax.plot([2, 5], [0.5, 0], color='#4ade80', linewidth=3, label='+3')
#             ax.scatter([0, 2, 5], [0, 0.5, 0], color='#f8fafc', s=60, zorder=5)
#             ax.set_xlim(-1, 7)
#             ax.set_ylim(-0.5, 1)
#             ax.set_yticks([])
#             ax.set_xticks(range(0, 7))
#             ax.set_title("San Okunda Goşmak (2 + 3 = 5)", fontsize=10)
#             ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', loc='upper right')
#         else:
#             labels = ['San 1', 'San 2', 'Jemi (Sum)']
#             values = [2, 3, 5]
#             colors = ['#38bdf8', '#a855f7', '#4ade80']
#             bars = ax.bar(labels, values, color=colors, width=0.5)
#             for bar in bars:
#                 yval = bar.get_height()
#                 ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, int(yval), ha='center', va='bottom', color='#f8fafc', fontweight='bold')
#             ax.set_ylim(0, 7)
#             ax.set_title("Goşulyjylar we Jemi", fontsize=10)
#         ax.grid(axis='y', color='#334155', linestyle=':', alpha=0.6)

#     elif any(w in full_topic for w in ["kalkulus", "calculus", "önüm", "töreme", "derivative", "integral"]):
#         x = np.linspace(-3, 3, 200)
#         if slide_index % 2 == 0:
#             ax.plot(x, x**2, color='#38bdf8', linewidth=2.5, label='f(x) = x²')
#             ax.plot(x, 2*x, color='#a855f7', linewidth=2, linestyle='--', label="f'(x) = 2x")
#             ax.set_title("Funksiýa we Onuň Önümi", fontsize=10)
#         else:
#             ax.plot(x, x**3 - 3*x, color='#f43f5e', linewidth=2.5, label='f(x) = x³ - 3x')
#             ax.set_title("Kübiki Funksiýa", fontsize=10)
#         ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', loc='upper left')
#         ax.grid(True, color='#334155', linestyle=':', alpha=0.6)

#     else:
#         x = np.linspace(1, 10, 100)
#         y = np.log(x) * 10
#         ax.plot(x, y, color='#a855f7', linewidth=2.5)
#         ax.fill_between(x, y, color='#a855f7', alpha=0.2)
#         ax.set_title("Sapak Boýunça Bilim Ösüşi", fontsize=10)
#         ax.grid(True, color='#334155', linestyle=':', alpha=0.6)

#     plt.tight_layout()
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor(), transparent=False)
#     plt.close(fig)
#     buf.seek(0)

#     image = QImage()
#     image.loadFromData(buf.getvalue())
#     return image


# def save_slide_images(slides, subject, topic, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     app = QApplication.instance() or QApplication(sys.argv)

#     for idx, slide in enumerate(slides):
#         img = QImage(1280, 720, QImage.Format.Format_ARGB32)
#         img.fill(QColor("#0f172a"))

#         painter = QPainter(img)
#         painter.setRenderHint(QPainter.RenderHint.Antialiasing)

#         painter.setPen(Qt.PenStyle.NoPen)
#         painter.setBrush(QBrush(QColor(30, 41, 59, 240)))
#         painter.drawRoundedRect(40, 40, 1200, 640, 20, 20)

#         painter.setBrush(QBrush(QColor(56, 189, 248)))
#         painter.drawRoundedRect(70, 70, 10, 45, 5, 5)

#         painter.setPen(QColor(248, 250, 252))
#         painter.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
#         painter.drawText(QRect(95, 68, 1100, 50), Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, slide["title"])

#         painter.setPen(QColor(203, 213, 225))
#         painter.setFont(QFont("Segoe UI", 16))
#         painter.drawText(QRect(95, 145, 650, 500), Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, slide["content"])

#         graph_img = generate_topic_graph(idx, subject, topic, slide["title"])
#         if not graph_img.isNull():
#             painter.drawImage(760, 150, graph_img)

#         painter.end()
#         img.save(os.path.join(output_dir, f"slide_{idx + 1}.png"))


# def generate_and_save_tts(text, output_wav_path, speed_factor=1.2):
#     sentences = re.split(r'(?<=[.!?])\s+', text)
#     chunks, current_chunk = [], ""

#     for s in sentences:
#         if len(current_chunk) + len(s) < 220:
#             current_chunk += " " + s
#         else:
#             if current_chunk.strip():
#                 chunks.append(current_chunk.strip())
#             current_chunk = s
#     if current_chunk.strip():
#         chunks.append(current_chunk.strip())

#     audio_segments = []
#     for chunk in chunks:
#         inputs = tokenizer(chunk, return_tensors="pt")
#         with torch.no_grad():
#             output = model(**inputs).waveform
#         segment = output.squeeze().cpu().numpy()
#         audio_segments.append(segment)

#     if not audio_segments:
#         return

#     full_audio = np.concatenate(audio_segments)

#     if speed_factor != 1.0:
#         indices = np.arange(0, len(full_audio), speed_factor)
#         full_audio = np.interp(indices, np.arange(len(full_audio)), full_audio).astype(np.float32)

#     # Convert float [-1.0, 1.0] to int16 PCM format
#     audio_int16 = (full_audio * 32767).clip(-32768, 32767).astype(np.int16)
    
#     os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
#     with wave.open(output_wav_path, 'wb') as wav_file:
#         wav_file.setnchannels(1)
#         wav_file.setsampwidth(2)
#         wav_file.setframerate(model.config.sampling_rate)
#         wav_file.writeframes(audio_int16.tobytes())


# def main():
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     db_path = os.path.join(base_dir, "database", "lessons.db")
#     output_dir = os.path.join(base_dir, "output")

#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
#     cursor.execute("SELECT subject, topic, duration_minutes FROM lessons ORDER BY id DESC LIMIT 1")
#     row = cursor.fetchone()
#     conn.close()

#     subject, topic, duration_minutes = row if row else ("Matematika", "Goşmak", 10)

#     api_key = os.getenv("GEMINI_API_KEY")
#     client = genai.Client(api_key=api_key)

#     target_slide_count = max(2, int(duration_minutes * 2))
#     target_word_count = max(600, int(duration_minutes * 600))

#     prompt = f"""
# Siz tejribeli mugallym. Sapagyň mowzugy: {subject} - {topic}.
# Sapagyň dowamlylygy: {duration_minutes} minut.

# Haýyş, ähli jogaby diňe Türkmen dilinde (Latyn elipbiýinde) doly, giňišleýin we düşnükli beriň.

# Wajyp düzgünler:
# 1. Sapagy örän gyzykly, özüne çekiji, janly we TÄSIRLI ediň, hiç hili içgysgyn bolmasyn! (Make really interesting rather than boring).
# 2. Sapagyň dowamlylygy {duration_minutes} minut bolany üçin hut {target_slide_count} sany slayd dörediň (her minut üçin 2 slayd).
# 3. FULL_SPEECH bölüminde edil {target_word_count} söz töweregi (her minut üçin 600 söz) giňišleýin düşündiriş ýazyň.
# 4. Gürrüňiň içinde minutlary asla agzamaň (meselem: "häzir 1-nji minutda", "2-nji minutdarys", "minut geçdi" diýip AÝTMAŇ!).
# 5. Riyazi formulalarda we simwollarda raw LaTeX ulanmaň! Onuň deregine ýönekeý Unicode simwollaryny ulanyň.

# Jogaby tapawutlandyrmak üçin edil ashakdaky ýaly strukturada ýazyň:

# SLIDE_1:
# Sözbaşy: [1-nji Slaydyň gysga sözbaşysy]
# Mazmuny:
# - [Täsirli we düşnükli esasy nokat]

# FULL_SPEECH:
# [Bu ýerde çagalara aýtjak takmynan {target_word_count} sözden ybarat bolan, örän gyzykly gürrüňiňizi ýazyň.]
# """

#     print("Generation started...")
#     response = client.models.generate_content(
#         model='gemini-3.5-flash-lite',
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
#         )
#     )

#     raw_text = response.text
#     slides, speech = [], ""

#     if "FULL_SPEECH:" in raw_text:
#         parts = raw_text.split("FULL_SPEECH:")
#         slide_part, speech = parts[0], parts[1].strip()
#     else:
#         slide_part, speech = raw_text, raw_text

#     for block in slide_part.split("SLIDE_")[1:]:
#         lines = [line.strip() for line in block.split("\n") if line.strip()]
#         title, content_lines = "Sapak", []
#         for line in lines:
#             if line.startswith("Sözbaşy:"):
#                 title = clean_latex_math(line.replace("Sözbaşy:", "").strip())
#             elif not line.startswith("Mazmuny:"):
#                 content_lines.append(clean_latex_math(line))
#         slides.append({"title": title, "content": "\n".join(content_lines)})

#     # Render Slides to output/
#     save_slide_images(slides, subject, topic, output_dir)

#     # Render TTS Audio to output/lesson_audio.wav
#     audio_wav_path = os.path.join(output_dir, "lesson_audio.wav")
#     generate_and_save_tts(clean_latex_math(speech), audio_wav_path, speed_factor=1.2)

#     print("Lesson generated successfully!")


# if __name__ == "__main__":
#     main()