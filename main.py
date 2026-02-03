import sys
import os
import cv2
import numpy as np
from datetime import datetime, date
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QListWidget,
    QMessageBox, QLineEdit, QTabWidget
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import xlwt
from xlutils.copy import copy as xl_copy
import xlrd


class SimpleFaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Attendance System")
        self.setGeometry(100, 100, 1000, 700)

        self.known_faces = []
        self.known_face_names = []
        self.attendance_recorded = []
        self.attendance_file = "attendence_excel.xls"
        self.images_folder = "known_faces"

        if not os.path.exists(self.images_folder):
            os.makedirs(self.images_folder)

        self.load_known_faces()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.video_capture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.init_ui()

    def load_known_faces(self):
        self.known_faces = []
        self.known_face_names = []

        for filename in os.listdir(self.images_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                path = os.path.join(self.images_folder, filename)
                image = cv2.imread(path)
                if image is not None:
                    name = os.path.splitext(filename)[0]
                    self.known_faces.append(image)
                    self.known_face_names.append(name)

        self.update_faces_list()

    def update_faces_list(self):
        if hasattr(self, "faces_list"):
            self.faces_list.clear()
            for name in self.known_face_names:
                self.faces_list.addItem(name)

    def compare_faces_simple(self, face_region):
        try:
            if not self.known_faces:
                return "Unknown"

            face_resized = cv2.resize(face_region, (100, 100))
            best_match = "Unknown"
            best_score = float("inf")

            for i, known_face in enumerate(self.known_faces):
                known_resized = cv2.resize(known_face, (100, 100))
                diff = cv2.absdiff(face_resized, known_resized)
                mse = np.mean(diff ** 2)

                if mse < best_score and mse < 5000:
                    best_score = mse
                    best_match = self.known_face_names[i]

            return best_match
        except:
            return "Unknown"

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                face_region = frame[y:y + h, x:x + w]
                name = self.compare_faces_simple(face_region)

                cv2.rectangle(frame, (x, y + h - 35), (x + w, y + h),
                              (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (x + 6, y + h - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

                if name != "Unknown" and name not in self.attendance_recorded:
                    self.attendance_recorded.append(name)
                    QMessageBox.information(
                        self, "Attendance",
                        f"{name} has been marked present"
                    )

            self.display_frame(frame)

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qt_image).scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.KeepAspectRatio
            )
        )

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        camera_tab = QWidget()
        camera_layout = QVBoxLayout(camera_tab)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid black;")
        camera_layout.addWidget(self.video_label)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start Recognition")
        self.stop_button = QPushButton("Stop Recognition")
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_recognition)
        self.stop_button.clicked.connect(self.stop_recognition)

        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        camera_layout.addLayout(buttons)

        add_face_tab = QWidget()
        add_layout = QVBoxLayout(add_face_tab)

        name_layout = QHBoxLayout()
        self.name_input = QLineEdit()
        capture_button = QPushButton("Capture Image")
        capture_button.clicked.connect(self.capture_image)

        name_layout.addWidget(QLabel("Name:"))
        name_layout.addWidget(self.name_input)
        name_layout.addWidget(capture_button)
        add_layout.addLayout(name_layout)

        self.faces_list = QListWidget()
        add_layout.addWidget(QLabel("Known Faces:"))
        add_layout.addWidget(self.faces_list)

        tabs.addTab(camera_tab, "Face Recognition")
        tabs.addTab(add_face_tab, "Add New Face")

    def capture_image(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a name")
            return

        ret, frame = self.video_capture.read()
        if ret:
            path = os.path.join(self.images_folder, f"{name}.png")
            cv2.imwrite(path, frame)
            QMessageBox.information(self, "Saved", "Image saved successfully")
            self.name_input.clear()
            self.load_known_faces()

    def start_recognition(self):
        self.timer.start(30)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_recognition(self):
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_attendance()

    def save_attendance(self):
        try:
            try:
                rb = xlrd.open_workbook(self.attendance_file, formatting_info=True)
                wb = xl_copy(rb)
                sheet = wb.add_sheet(date.today().strftime("%Y-%m-%d"))
            except:
                wb = xlwt.Workbook()
                sheet = wb.add_sheet(date.today().strftime("%Y-%m-%d"))

            sheet.write(0, 0, "Name")
            sheet.write(0, 1, "Status")
            sheet.write(0, 2, "Date")
            sheet.write(0, 3, "Time")

            for i, name in enumerate(self.attendance_recorded, 1):
                sheet.write(i, 0, name)
                sheet.write(i, 1, "Present")
                sheet.write(i, 2, date.today().strftime("%Y-%m-%d"))
                sheet.write(i, 3, datetime.now().strftime("%H:%M:%S"))

            wb.save(self.attendance_file)
            QMessageBox.information(self, "Saved", "Attendance saved successfully")

        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def closeEvent(self, event):
        self.timer.stop()
        self.video_capture.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimpleFaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
