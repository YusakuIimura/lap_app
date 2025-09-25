# main_window.py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QDateEdit, QPushButton,
    QCheckBox, QListWidget, QListWidgetItem, QHBoxLayout, QMessageBox, QDateTimeEdit
)
from PyQt5.QtCore import QDateTime
from kpi_page import KPIPage
import sys
from utils import get_df_from_db
from utils import get_df_from_db, jst_str_to_utc_sql
import os
import json
import time

# setting読み取り
SETTINGS_PATH = os.path.join(os.getcwd(), "settings.json")

def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

class MainWindow(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.start_datetime = QDateTimeEdit()
        self.end_datetime = QDateTimeEdit()

        self.setWindowTitle("Lap Time Visualizer")
        self.resize(1200, 800)
        
        # 設定ロード＆初期値適用
        self._settings = _load_json(SETTINGS_PATH)
        ui_cfg = self._settings.get("ui", {})
        sd = ui_cfg.get("start_datetime")
        ed = ui_cfg.get("end_datetime")
        if isinstance(sd, str):
            dt = QDateTime.fromString(sd, "yyyy-MM-dd HH:mm:ss")
            if dt.isValid():
                self.start_datetime.setDateTime(dt)
        if isinstance(ed, str):
            dt = QDateTime.fromString(ed, "yyyy-MM-dd HH:mm:ss")
            if dt.isValid():
                self.end_datetime.setDateTime(dt)

        self.layout = QVBoxLayout()


        # self.start_datetime.setDateTime(QDateTime.fromString("2025-06-29 00:00:00", "yyyy-MM-dd HH:mm:ss"))
        # self.end_datetime.setDateTime(QDateTime.fromString("2025-06-30 23:59:59", "yyyy-MM-dd HH:mm:ss"))
        self.start_datetime.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.end_datetime.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        
        self.layout.addWidget(QLabel("Start Date"))
        self.layout.addWidget(self.start_datetime)
        self.layout.addWidget(QLabel("End Date"))
        self.layout.addWidget(self.end_datetime)

        self.search_btn = QPushButton("Get players list")
        self.search_btn.clicked.connect(self.load_players)
        self.layout.addWidget(self.search_btn)

        self.player_list = QListWidget()
        self.player_list.setSelectionMode(QListWidget.MultiSelection)
        self.layout.addWidget(self.player_list)

        self.kpi_btn = QPushButton("Show KPIs")
        self.kpi_btn.clicked.connect(self.show_kpi)
        self.layout.addWidget(self.kpi_btn)

        self.setLayout(self.layout)
        
        # 変更されたら即保存
        self.start_datetime.dateTimeChanged.connect(self._persist_dates)
        self.end_datetime.dateTimeChanged.connect(self._persist_dates)

    def load_players(self):
        start_jst = self.start_datetime.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        end_jst   = self.end_datetime.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        start_utc = jst_str_to_utc_sql(start_jst)
        end_utc   = jst_str_to_utc_sql(end_jst)
        query = f"""
        SELECT DISTINCT u.first_name, u.last_name, u.id
        FROM passing p
        JOIN transponder_user tu ON p.transponder_id = tu.transponder_id
        JOIN user u ON tu.user_id = u.id
        WHERE p.timestamp BETWEEN '{start_utc}' AND '{end_utc}'
        AND p.timestamp BETWEEN tu.since AND tu.until
        ORDER BY u.last_name, u.first_name;
        """
        print("[Players] クエリ送信中…")
        t0 = time.perf_counter()
        df = get_df_from_db(query)
        t1 = time.perf_counter()
        print(f"[Players] 取得完了: {len(df)} 行 / {t1 - t0:.2f}s")
        
        self.player_list.clear()
        if df.empty:
           QMessageBox.information(self, "No players", "No players found for the specified period. Please change the period and search again.")
           return
        for _, row in df.iterrows():
            name = f"{row['last_name']} {row['first_name']} ({row['id']})"
            item = QListWidgetItem(name)
            item.setData(1, row['id'])
            self.player_list.addItem(item)

    def show_kpi(self):
        selected_ids = [item.data(1) for item in self.player_list.selectedItems()]
        if not selected_ids:
            QMessageBox.warning(self, "Player Unselected", "Please select at least one player")
            return

        start_jst = self.start_datetime.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        end_jst   = self.end_datetime.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        start_utc = jst_str_to_utc_sql(start_jst)
        end_utc   = jst_str_to_utc_sql(end_jst)
        ids_str = ",".join(map(str, selected_ids))

        query = f"""
        SELECT 
            p.timestamp,
            p.decoder_id,
            u.first_name,
            u.last_name,
            p.transponder_id,
            tu.user_id
        FROM 
            passing p
        JOIN (
            SELECT id, transponder_id, user_id
            FROM transponder_user
            WHERE user_id IN ({ids_str})
            AND since <= '{start_utc}'
            AND (until IS NULL OR until > '{start_utc}')
        ) tu
        ON tu.transponder_id = p.transponder_id
        JOIN `user` u ON u.id = tu.user_id
        WHERE 
            p.timestamp BETWEEN '{start_utc}' AND '{end_utc}'
            AND tu.user_id IN ({ids_str})
        ORDER BY 
            p.timestamp
        LIMIT 10000;
        """

        # ページ遷移
        kpi_page = KPIPage(query, self.stacked_widget)
        self.stacked_widget.addWidget(kpi_page)
        self.stacked_widget.setCurrentWidget(kpi_page)

    def _persist_dates(self):
        sd = self.start_datetime.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        ed = self.end_datetime.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        self._settings.setdefault("ui", {})
        self._settings["ui"]["start_datetime"] = sd
        self._settings["ui"]["end_datetime"] = ed
        _save_json(SETTINGS_PATH, self._settings)
