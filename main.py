# main.py
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
import sys

from image_ui import ImageSuite
from audio_ui import AudioSuite
from video_ui import VideoSuite


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stego Studio (Image, Audio & Video)")
        self.resize(1220, 800)

        self.top_tabs = QTabWidget()
        self.top_tabs.addTab(ImageSuite(), "Image")
        self.top_tabs.addTab(AudioSuite(), "Audio")
        self.top_tabs.addTab(VideoSuite(), "Video")
        self.setCentralWidget(self.top_tabs)

        self._build_menu()
        self.statusBar().showMessage("Ready")

    def _build_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        act_quit = QAction("Exit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
