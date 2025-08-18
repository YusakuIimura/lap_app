from PyQt5.QtWidgets import QApplication, QStackedWidget
from main_window import MainWindow
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # ページスタック
    stacked = QStackedWidget()

    main_page = MainWindow(stacked)
    stacked.addWidget(main_page) # index 0

    stacked.setCurrentIndex(0)
    stacked.resize(1200, 800)
    stacked.show()

    sys.exit(app.exec_())
