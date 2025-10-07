
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QComboBox, QTextEdit, QFrame, QRadioButton, QButtonGroup, QFileDialog
)
from PyQt5.QtCore import Qt

class Ui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Parking — License Plate OCR")
        self.setMinimumSize(1200, 750)
        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        # Video preview
        self.label_main = QLabel("Camera / Video")
        self.label_main.setAlignment(Qt.AlignCenter)
        self.label_main.setMinimumSize(800, 450)
        self.label_main.setFrameShape(QFrame.Box)

        # Plate crop
        self.label_plate_img = QLabel("Plate ROI")
        self.label_plate_img.setAlignment(Qt.AlignCenter)
        self.label_plate_img.setMinimumSize(300, 120)
        self.label_plate_img.setFrameShape(QFrame.Box)

        # Text fields
        self.label_digits = QLabel("—")
        self.label_time   = QLabel("—")
        self.label_price  = QLabel("—")
        self.label_status = QLabel("—")

        for l in (self.label_digits, self.label_time, self.label_price, self.label_status):
            l.setAlignment(Qt.AlignCenter)
            l.setFrameShape(QFrame.Box)
            l.setMinimumHeight(48)

        # Buttons
        self.btn_open_video = QPushButton("Open Video…")
        self.btn_use_camera = QPushButton("Use Camera 0")
        self.btn_check_in   = QPushButton("Check in")
        self.btn_check_out  = QPushButton("Check out")
        self.btn_resume = QPushButton("Tiếp tục")

        # Vehicle type
        self.radio_car   = QRadioButton("Ô tô (5000 VND)")
        self.radio_bike  = QRadioButton("Xe máy (3000 VND)")
        self.radio_bike.setChecked(True)
        self.group_vehicle = QButtonGroup()
        self.group_vehicle.addButton(self.radio_bike)
        self.group_vehicle.addButton(self.radio_car)

        # Log pane
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setMinimumHeight(120)

        # Right panel grid
        grid = QGridLayout()
        grid.addWidget(QLabel("Biển đọc được:"), 0,0)
        grid.addWidget(self.label_digits,       0,1)
        grid.addWidget(QLabel("Ảnh biển số:"),  1,0)
        grid.addWidget(self.label_plate_img,    1,1)
        grid.addWidget(QLabel("Thời gian vào:"),2,0)
        grid.addWidget(self.label_time,         2,1)
        grid.addWidget(QLabel("Trạng thái:"),   3,0)
        grid.addWidget(self.label_status,       3,1)
        grid.addWidget(QLabel("Giá vé:"),       4,0)
        grid.addWidget(self.label_price,        4,1)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.addWidget(self.btn_open_video)
        ctrl.addWidget(self.btn_use_camera)
        ctrl.addStretch(1)
        ctrl.addWidget(self.radio_bike)
        ctrl.addWidget(self.radio_car)
        ctrl.addStretch(1)
        ctrl.addWidget(self.btn_check_in)
        ctrl.addWidget(self.btn_check_out)
        ctrl.addWidget(self.btn_resume)


        # Main layout
        root = QVBoxLayout(self)
        root.addWidget(self.label_main, 3)
        root.addLayout(grid, 2)
        root.addLayout(ctrl)
        root.addWidget(self.text_log, 1)

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget { background: #0f172a; color: #e2e8f0; font-size: 14px; }
            QLabel#title { font-size: 24px; font-weight: 700; }
            QLabel { border-radius: 8px; }
            QFrame { border: 2px solid #334155; border-radius: 12px; }
            QPushButton {
                background: #1e293b; border: 1px solid #334155; padding: 10px 16px;
                border-radius: 10px;
            }
            QPushButton:hover { background: #0b1220; }
            QRadioButton { padding: 0 8px; }
            QTextEdit { background: #0b1220; border: 1px solid #334155; border-radius: 10px; }
        """)
