
import sys, cv2, numpy as np, os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from ui_main import Ui
from ocr_plate import OcrPlate
from check_and_save_img import CheckAndSaveImg

# --- Config ---
PLATE_WEIGHTS = "model/best.pt"         # seg/det plate model (.pt or .onnx)
CHAR_WEIGHTS  = "model/best_ocr.pt"  # char det model (.pt or .onnx)
# --- Auto-capture config ---
PLATE_MIN_SCORE   = 0.8   # điểm tối thiểu để chấp nhận auto chụp
NO_PLATE_FRAMES   = 12     # số frame liên tiếp không thấy biển => chốt & chụp
COOLDOWN_FRAMES   = 40     # sau khi chụp xong, bỏ qua bấy nhiêu frame để tránh chụp lặp
FRAME_STRIDE      = 3      # xử lý 1 frame / 3 cho đỡ lag

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui()
        self.setCentralWidget(self.ui)
        self.setWindowTitle("Smart Parking — License Plate OCR")

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._loop)

        self.ocr = OcrPlate(PLATE_WEIGHTS, CHAR_WEIGHTS, det_conf_plate=0.25, det_conf_char=0.30)
        self.store = CheckAndSaveImg()

        self.paused = False


        # auto-capture state
        self.frame_id   = 0
        self.cooldown   = 0
        self.no_plate   = 0
        self.best = {"plate": None, "score": 0.0, "frame": None}  # khung hình gốc đẹp nhất


        # state
        self.current_digits = None
        self.current_roi = None

        # wire UI
        self.ui.btn_use_camera.clicked.connect(self._open_camera)
        self.ui.btn_open_video.clicked.connect(self._open_video_file)
        self.ui.btn_check_in.clicked.connect(self._pause_video)
        self.ui.btn_resume.clicked.connect(self._resume_video)

        self.ui.btn_check_out.clicked.connect(self._check_out)
        

        self._open_camera()

    # --- Video IO ---
    def _open_camera(self):
        self._release_cap()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._log("Không thể mở camera 0")
        else:
            self._log("Đang dùng camera 0")
            self.timer.start(15)

    def _open_video_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn video", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self._release_cap()
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                self._log(f"Không mở được video: {path}")
            else:
                self._log(f"Phát video: {os.path.basename(path)}")
                self.timer.start(15)

    def _release_cap(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.timer.stop()

    def _loop(self):
        if self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self._log("Hết video / mất kết nối camera")
            self.timer.stop()
            return

        self.frame_id = getattr(self, "frame_id", 0) + 1
        if self.frame_id % 2 != 0:  # chỉ xử lý 1/5 frame
            return


        if self.cooldown > 0:
            self.cooldown -= 1

        annotated, text, score, roi = self.ocr.infer_image(frame)
        self._show_frame(self.ui.label_main, annotated)
        if roi is not None:
            self._show_frame(self.ui.label_plate_img, roi)

        # cập nhật UI cơ bản
        self.current_digits = None if text == "unknown" else text
        if self.current_digits:
            self.ui.label_digits.setText(self.current_digits)
            price = 3000 if self.ui.radio_bike.isChecked() else 5000
            self.ui.label_price.setText(f"{price:,} VND")
        else:
            for w in (self.ui.label_digits, self.ui.label_time, self.ui.label_status, self.ui.label_price):
                w.setText("—")

        # # ====== AUTO-CAPTURE LOGIC ======
        # if self.current_digits and score >= 0.01:  # có biển
        #     self.no_plate = 0

        #     # nếu đang cooldown thì không update best (tránh chụp lặp)
        #     if self.cooldown == 0:
        #         # nếu cùng 1 biển với best -> cập nhật nếu điểm cao hơn
        #         if self.best["plate"] == self.current_digits:
        #             if score > self.best["score"]:
        #                 self.best.update({"score": score, "frame": frame.copy()})
        #         # nếu chưa có best -> set
        #         elif self.best["plate"] is None:
        #             self.best = {"plate": self.current_digits, "score": score, "frame": frame.copy()}
        #         # nếu biển đổi sang cái khác -> chốt cái cũ (nếu đạt ngưỡng), rồi bắt đầu theo dõi biển mới
        #         else:
        #             self._maybe_commit_best()
        #             self.best = {"plate": self.current_digits, "score": score, "frame": frame.copy()}
        # else:
        #     # không thấy biển trong frame này
        #     self.no_plate += 1
        #     # khi chuỗi biển vừa rời khung đủ lâu -> chốt
        #     if self.no_plate >= NO_PLATE_FRAMES:
        #         self._maybe_commit_best()

        # ====== Hiển thị status hiện tại trong bãi ======
        if self.current_digits and self.store.check_exists(self.current_digits):
            t, _ = self.store.get_data(self.current_digits)
            self.ui.label_time.setText(t if isinstance(t, str) else str(t))
            self.ui.label_status.setText("Đã trong bãi")
        elif self.current_digits:
            self.ui.label_status.setText("Chưa vào bãi")


    def _pause_video(self):
        """Khi bấm Check in thì tạm dừng video và lưu ảnh."""
        if not self.cap:
            self._log(" Chưa mở video hoặc camera.")
            return

        if not self.paused:
            self.paused = True
            self.timer.stop()

            #  Chụp và lưu ảnh tại thời điểm dừng
            ok, frame = self.cap.read()
            if ok and self.current_digits:
                self.store.save_image(self.current_digits, frame)
                from datetime import datetime
                now = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
                self.ui.label_status.setText("Đã vào bãi")
                self.ui.label_time.setText(now)
                self._log(f" Đã lưu ảnh biển {self.current_digits} lúc {now}")
            else:
                self._log(" Không có biển hoặc không lấy được khung hình để lưu.")
            self._log(" Video đã tạm dừng để Check-in.")
        else:
            self._log(" Video đang bị tạm dừng rồi.")


    def _resume_video(self):
        """Tiếp tục video sau khi tạm dừng."""
        if self.paused:
            self.paused = False
            self.timer.start(15)
            self._log(" Tiếp tục phát video.")
        else:
            self._log("Video đang chạy rồi.")




    # # --- Actions ---
    # def _check_in(self):
    #     if not self.current_digits:
    #         self._log("Chưa nhận dạng được biển để check-in")
    #         return
    #     if self.store.check_exists(self.current_digits):
    #         self._log(f"{self.current_digits} đã trong bãi")
    #         return
    #     # Save current frame as entry image
    #     ok, frame = self.cap.read()
    #     if ok:
    #         self.store.save_image(self.current_digits, frame)
    #         self._log(f"Đã lưu ảnh vào bãi: {self.current_digits}")
    #     else:
    #         self._log("Không lấy được khung hình để lưu")

    def _check_out(self):
        if not self.current_digits:
            self._log("Chưa nhận dạng được biển để check-out")
            return
        if not self.store.check_exists(self.current_digits):
            self._log(f"Không tìm thấy {self.current_digits} trong bãi")
            return
        self.store.delete_image(self.current_digits)
        self._log(f"Đã check-out và xoá: {self.current_digits}")
        self.ui.label_status.setText("—")
        self.ui.label_time.setText("—")

    # --- Helpers ---
    def _show_frame(self, label, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(), aspectRatioMode=1))

    def _log(self, msg: str):
        self.ui.text_log.append(msg)

    def closeEvent(self, event):
        self._release_cap()
        event.accept()
"""
    def _maybe_commit_best(self):
        Chụp ảnh auto nếu best đạt ngưỡng, rồi reset & cooldown.
        if self.best["plate"] and self.best["score"] >= PLATE_MIN_SCORE and self.best["frame"] is not None:
            plate = self.best["plate"]
            self.store.save_image(plate, self.best["frame"])
            self._log(f"[AUTO] Captured {plate} (score={self.best['score']:.2f})")
            self.cooldown = COOLDOWN_FRAMES
        # reset best & counters
        self.best = {"plate": None, "score": 0.0, "frame": None}
        self.no_plate = 0
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    m = Main()
    m.show()
    sys.exit(app.exec_())
