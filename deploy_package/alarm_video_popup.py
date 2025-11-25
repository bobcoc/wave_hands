import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QSlider, QHBoxLayout, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class AlarmVideoPopup(QMainWindow):
    def __init__(self, video_path, window_name='ALARM', parent=None):
        super().__init__(parent)
        self.setWindowTitle(window_name)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0
        self.is_playing = False

        # UI
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.play_btn = QPushButton('播放')
        self.pause_btn = QPushButton('暂停')
        self.close_btn = QPushButton('关闭')
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.total_frames-1)
        self.slider.setValue(0)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.play_btn)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.close_btn)
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label)
        main_layout.addWidget(self.slider)
        main_layout.addLayout(btn_layout)
        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        # Connect
        self.play_btn.clicked.connect(self.start_play)
        self.pause_btn.clicked.connect(self.pause_play)
        self.close_btn.clicked.connect(self.close_window)
        self.slider.valueChanged.connect(self.seek_frame)

        # 初始显示第一帧
        self.show_frame(0)
        self.pause_play()

    def show_frame(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)
        self.current_frame = frame_idx

    def next_frame(self):
        if self.current_frame < self.total_frames-1:
            self.current_frame += 1
            self.show_frame(self.current_frame)
        else:
            self.pause_play()

    def start_play(self):
        self.is_playing = True
        self.timer.start(int(1000/(self.fps if self.fps>0 else 25)))

    def pause_play(self):
        self.is_playing = False
        self.timer.stop()

    def seek_frame(self, value):
        self.show_frame(value)

    def close_window(self):
        self.close()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()

def show_alarm_video_popup(video_path, window_name='ALARM'):
    app = QApplication(sys.argv)
    win = AlarmVideoPopup(video_path, window_name)
    win.show()
    app.exec_() 