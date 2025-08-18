# main_window.py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QDateEdit, QPushButton,
    QCheckBox, QListWidget, QListWidgetItem, QHBoxLayout, QMessageBox, QDateTimeEdit
)
from PyQt5.QtCore import QDateTime
from kpi_page import KPIPage
import sys
import mysql.connector
from utils import get_df_from_db

class MainWindow(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        self.setWindowTitle("Lap Time Visualizer")
        self.resize(1200, 800)

        self.layout = QVBoxLayout()

        self.start_datetime = QDateTimeEdit()
        self.end_datetime = QDateTimeEdit()
        self.start_datetime.setDateTime(QDateTime.fromString("2025-06-29 00:00:00", "yyyy-MM-dd HH:mm:ss"))
        self.end_datetime.setDateTime(QDateTime.fromString("2025-06-30 23:59:59", "yyyy-MM-dd HH:mm:ss"))
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

    def load_players(self):
        start = self.start_datetime.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        end = self.end_datetime.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        query = f"""
        SELECT DISTINCT u.first_name, u.last_name, u.id
        FROM passing p
        JOIN transponder_user tu ON p.transponder_id = tu.transponder_id
        JOIN user u ON tu.user_id = u.id
        WHERE p.timestamp BETWEEN '{start}' AND '{end}'
        AND p.timestamp BETWEEN tu.since AND tu.until
        ORDER BY u.last_name, u.first_name;
        """
        df = get_df_from_db(query)
        self.player_list.clear()
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

        start = self.start_datetime.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        end = self.end_datetime.dateTime().toString("yyyy-MM-dd HH:mm:ss")
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
        JOIN 
            transponder_user tu ON p.transponder_id = tu.transponder_id
        JOIN 
            user u ON tu.user_id = u.id
        WHERE 
            p.timestamp BETWEEN '{start}' AND '{end}'
            AND tu.user_id IN ({ids_str})
        ORDER BY 
            p.timestamp
        LIMIT 10000;
        """

        # ページ遷移
        kpi_page = KPIPage(query, self.stacked_widget)
        self.stacked_widget.addWidget(kpi_page)
        self.stacked_widget.setCurrentWidget(kpi_page)
