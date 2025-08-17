from PyQt5.QtWidgets import QApplication, QStackedWidget
from main_window import MainWindow
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 1つのウィンドウでページを切替えるためのスタック
    stacked = QStackedWidget()

    # MainWindow を作ってスタックに追加
    main_page = MainWindow(stacked)
    stacked.addWidget(main_page)           # index 0

    # 初期表示は MainWindow
    stacked.setCurrentIndex(0)
    stacked.resize(800, 600)
    stacked.show()

    sys.exit(app.exec_())
