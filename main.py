from main_window import MainWindow
import sys
import os
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QStackedWidget, QWidget, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QMessageBox, QFileDialog,QDialog
)
from PyQt5.QtGui import QFont
import json

ENFORCE_LICENSE_AFTER = "2024-12-31"
# ライセンスキーと期限の辞書
VALID_KEYS = {
    "fS4T9Pai3Qsu": "2025-12-31",
    "24wux3bDftRu": "2026-12-31"
}

# exe と同階層に置かれることを期待するライセンスファイル名
DEFAULT_LICENSE_FILE = "license.txt"

class LicenseDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("License Authentication")
        self.setFixedSize(300, 120)

        self.label = QLabel("Enter your license key:")
        self.input = QLineEdit()
        self.button = QPushButton("Verify")
        self.button.clicked.connect(self.verify_from_input)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.input)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def verify_key(self, key: str) -> bool:
        key = key.strip()
        expiry_str = VALID_KEYS.get(key)

        if not expiry_str:
            QMessageBox.critical(self, "Error", "Invalid license key.")
            return False

        try:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid expiry date format.")
            return False

        if datetime.now() > expiry_date:
            QMessageBox.critical(self, "Error", "This license key has expired.")
            return False

        self.accept()
        return True

    def verify_from_input(self):
        key = self.input.text()
        self.verify_key(key)

    def try_license_file(self, filepath: str) -> bool:
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    key = f.read().strip()
                    return self.verify_key(key)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to read license file: {str(e)}")
        return False

    def prompt_for_license_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select License File", "", "Text Files (*.txt)")
        if file_path:
            self.try_license_file(file_path)

    def accept(self):
        # QDialog.Accepted を返すだけ（MainWindowは main 側で開く）
        super().accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    SETTINGS_PATH = os.path.join(os.getcwd(), "settings.json")
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        ui = (cfg or {}).get("ui", {})
        family = ui.get("font_family", "Meiryo")
        size   = int(ui.get("font_size", 12))
        app.setFont(QFont(family, size))
    except Exception:
        # 読めない場合はデフォルトで少し大きめ
        app.setFont(QFont("Meiryo", 12))
    
    now = datetime.now()
    enforce_date = datetime.strptime(ENFORCE_LICENSE_AFTER, "%Y-%m-%d")

    if now < enforce_date:
        # 認証不要期間
        stacked = QStackedWidget()
        window = MainWindow(stacked)
        stacked.addWidget(window)
        stacked.setCurrentWidget(window)
        stacked.resize(1200, 800)
        stacked.show()
    else:
        # 認証が必要
        dlg = LicenseDialog()
        # まず自動認証（license.txt）。成功すれば accept() が呼ばれて Accepted 扱い。
        accepted = dlg.try_license_file(DEFAULT_LICENSE_FILE)
        # 自動でダメならモーダル表示して結果で判定
        if not accepted:
            accepted = (dlg.exec_() == QDialog.Accepted)
        if not accepted:
            # 失敗 or ダイアログを閉じた → アプリ終了
            sys.exit(0)


    # ページスタック
    stacked = QStackedWidget()

    main_page = MainWindow(stacked)
    stacked.addWidget(main_page) # index 0

    stacked.setCurrentIndex(0)
    stacked.resize(1200, 800)
    stacked.show()

    sys.exit(app.exec_())
