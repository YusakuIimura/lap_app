from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QHBoxLayout, QTableView, QFormLayout, QCheckBox,
    QAbstractItemView, QMessageBox, QFileDialog, QRadioButton, QButtonGroup, QHeaderView,
    QMenu, QAction, QComboBox,QTableWidget, QTableWidgetItem, QDialog
)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, QPoint
from PyQt5.QtGui import QBrush, QColor, QGuiApplication, QKeySequence, QPixmap

import math
import datetime
import numpy as np
import pandas as pd
import json
import os
import re

from utils import get_df_from_db, to_jst_naive, translate_dict

# --------------------------------------------------------------------------------------
# 共通定数 / ヘルパ
# --------------------------------------------------------------------------------------

SETTINGS_PATH = os.path.join(os.getcwd(), "settings.json")
KPI_INTERVALS_PATH = os.path.join(os.getcwd(), "kpi.json")

TRACK_ORDER = [
    "FP",
    "SB1",
    "0m",
    "60m",
    "AP1",
    "50m",
    "100m",
    "BP",
    "150m",
    "AP2",
    "200m",
]
# 名前系は常に先頭に出したい列
NAME_COLUMNS = ["first_name", "last_name", "Date", "FP"]


def _load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------------------
# DataFrame -> Qt Model
# --------------------------------------------------------------------------------------

class DataFrameModel(QAbstractTableModel):
    """
    単純な DataFrame 表示モデル。
    - Timestamp: HH:MM:SS(.mmm)
    - Timedelta: MM:SS.mmm
    - それ以外: str(val)
    """

    def __init__(self, df: pd.DataFrame, mask: pd.DataFrame | None = None):
        super().__init__()
        self._df = df.reset_index(drop=True)
        self._mask = mask

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        val = self._df.iat[index.row(), index.column()]

        if role == Qt.DisplayRole:
            if pd.isna(val):
                return ""

            # Timestamp 系
            if isinstance(val, (pd.Timestamp, datetime.datetime)):
                ms = val.microsecond // 1000
                fmt = "%H:%M:%S" if ms == 0 else "%H:%M:%S.%f"
                text = val.strftime(fmt)
                return text if ms == 0 else text[:-3]

            # Timedelta 系
            if isinstance(val, pd.Timedelta):
                total_ms = int(val / pd.Timedelta(milliseconds=1))
                sign = "-" if total_ms < 0 else ""
                total_ms = abs(total_ms)
                minutes, rem = divmod(total_ms, 60_000)
                seconds, ms = divmod(rem, 1000)
                return f"{sign}{minutes:02d}:{seconds:02d}.{ms:03d}"

            return str(val)

        # 補完セルなら赤字
        if role == Qt.ForegroundRole and self._mask is not None:
            try:
                if bool(self._mask.iat[index.row(), index.column()]):
                    return QBrush(QColor(220, 0, 0))
            except Exception:
                pass

        if role == Qt.EditRole:
            return "" if pd.isna(val) else val

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._df.columns[section]
        return str(section + 1)


class RangeFilterProxy(QSortFilterProxyModel):
    """
    単一KPIに対する数値レンジフィルタ。
    - visible_columns: 表示されている列名の順序
    - filter_col: 絞り込み対象列名（None ならフィルタ無し）
    - vmin, vmax: 数値レンジ（両方 None ならフィルタ無し）
    """

    def __init__(self, visible_columns: list[str]):
        super().__init__()
        self.visible_columns = list(visible_columns)
        self.filter_col: str | None = None
        self.vmin: float | None = None
        self.vmax: float | None = None

    def set_visible_columns(self, visible_columns: list[str]):
        self.visible_columns = list(visible_columns)
        self.invalidateFilter()

    def set_filter_column(self, col_name: str | None):
        self.filter_col = col_name
        self.invalidateFilter()

    def set_range(self, vmin: float | None, vmax: float | None):
        self.vmin = vmin
        self.vmax = vmax
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()

        # フィルタ列が指定されていない／レンジが無い場合は全通し
        if not self.filter_col or (self.vmin is None and self.vmax is None):
            return True

        try:
            col_idx = self.visible_columns.index(self.filter_col)
        except ValueError:
            return True  # 表示されていなければフィルタしない

        idx = model.index(source_row, col_idx)
        text = model.data(idx, Qt.DisplayRole)
        if text in (None, ""):
            return False

        try:
            val = float(text)
        except Exception:
            return False

        if self.vmin is not None and val < self.vmin:
            return False
        if self.vmax is not None and val > self.vmax:
            return False

        return True


class KPIJsonEditorPage(QWidget):
    """
    kpi.json をGUIで編集するページ。

    上: トラック図の画像
    下: モード(rolling/standing/flying)ごとの start/end 区間の一覧と追加・削除UI
    """

    def __init__(self, kpi_page: "KPIPage"):
        super().__init__()
        self.kpi_page = kpi_page
        self.stacked_widget = kpi_page.stacked_widget

        # 元の設定をコピーして編集用に保持
        import copy
        self._config = copy.deepcopy(kpi_page._interval_config) or {}
        for key in ("rolling", "standing", "flying"):
            self._config.setdefault(key, [])

        # start/end で選べる地点（TRACK_ORDERをそのまま使用）
        self.available_points = list(TRACK_ORDER)

        self._build_ui()
        self._refresh_mode_entries()

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.setWindowTitle("Edit KPI")
        main = QVBoxLayout()

        # 画像エリア
        img_label = QLabel()
        path = getattr(self.kpi_page, "track_image_path", "")
        if path and os.path.exists(path):
            pix = QPixmap(path)
            if not pix.isNull():
                pix = pix.scaledToWidth(900, Qt.SmoothTransformation)
                img_label.setPixmap(pix)
                img_label.setAlignment(Qt.AlignCenter)
            else:
                img_label.setText(f"Cannot load image: {path}")
        else:
            img_label.setText(f"Track image not found: {path}")
        main.addWidget(img_label)

        # モード選択
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Rolling", userData="rolling")
        self.mode_combo.addItem("Standing", userData="standing")
        self.mode_combo.addItem("Flying", userData="flying")
        self.mode_combo.currentIndexChanged.connect(self._refresh_mode_entries)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        main.addLayout(mode_row)

        # 現在の定義一覧テーブル
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Display Name", "start", "end"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        main.addWidget(self.table)

        # 追加用UI
        add_row = QHBoxLayout()
        self.start_combo = QComboBox()
        self.start_combo.addItems(self.available_points)
        self.end_combo = QComboBox()
        self.end_combo.addItems(self.available_points)
        
        # startのオフセット用ドロップダウン（次の周回、次の次の周回など）
        self.start_offset_combo = QComboBox()
        for i in range(11):  # 0から10まで
            self.start_offset_combo.addItem(str(i), userData=i)
        
        # endのオフセット用ドロップダウン（次の周回、次の次の周回など）
        self.end_offset_combo = QComboBox()
        for i in range(11):  # 0から10まで
            self.end_offset_combo.addItem(str(i), userData=i)
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Display Name（start-end if empty）")

        add_row.addWidget(QLabel("start:"))
        add_row.addWidget(self.start_combo)
        add_row.addWidget(QLabel("start offset:"))
        add_row.addWidget(self.start_offset_combo)
        add_row.addWidget(QLabel("end:"))
        add_row.addWidget(self.end_combo)
        add_row.addWidget(QLabel("end offset:"))
        add_row.addWidget(self.end_offset_combo)
        add_row.addWidget(QLabel("Display Name:"))
        add_row.addWidget(self.name_edit)

        self.btnAdd = QPushButton("Add")
        self.btnAdd.clicked.connect(self._on_add_clicked)
        add_row.addWidget(self.btnAdd)

        main.addLayout(add_row)
        
        # Usage comment
        comment_label = QLabel("Note: offset specifies which lap to use. Example: start=0m, offset=1 → next lap's 0m")
        comment_label.setWordWrap(True)
        comment_label.setStyleSheet("color: gray; font-size: 10pt;")
        main.addWidget(comment_label)

        # 操作用ボタン
        btn_row = QHBoxLayout()
        self.btnDelete = QPushButton("Delete Selected Row")
        self.btnDelete.clicked.connect(self._on_delete_clicked)
        btn_row.addWidget(self.btnDelete)

        btn_row.addStretch()

        self.btnCancel = QPushButton("Cancel")
        self.btnCancel.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self.btnCancel)

        self.btnSave = QPushButton("Save and Back")
        self.btnSave.clicked.connect(self._on_save_clicked)
        btn_row.addWidget(self.btnSave)

        main.addLayout(btn_row)

        self.setLayout(main)

    # ------------------------------------------------------------------
    # モードごとの一覧表示
    # ------------------------------------------------------------------
    def _current_mode(self) -> str:
        data = self.mode_combo.currentData()
        return data or "rolling"

    def _refresh_mode_entries(self):
        mode = self._current_mode()
        entries = self._config.get(mode, [])

        self.table.setRowCount(0)
        for ent in entries:
            start = ent.get("start", "")
            end = ent.get("end", "")
            name = ent.get("name") or f"{start}-{end}"
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(name)))
            self.table.setItem(row, 1, QTableWidgetItem(str(start)))
            self.table.setItem(row, 2, QTableWidgetItem(str(end)))

    # ------------------------------------------------------------------
    # 追加 / 削除 / 保存 / 戻る
    # ------------------------------------------------------------------
    def _on_add_clicked(self):
        mode = self._current_mode()
        start_base = self.start_combo.currentText()
        start_offset = self.start_offset_combo.currentData()
        end_base = self.end_combo.currentText()
        end_offset = self.end_offset_combo.currentData()
        name = self.name_edit.text().strip()

        if not start_base or not end_base:
            QMessageBox.warning(self, "Cannot add.", "Please select both start and end.")
            return

        # startにオフセットを追加（+0の場合は省略）
        if start_offset and start_offset > 0:
            start = f"{start_base}+{start_offset}"
        else:
            start = start_base

        # endにオフセットを追加（+0の場合は省略）
        if end_offset and end_offset > 0:
            end = f"{end_base}+{end_offset}"
        else:
            end = end_base

        entry = {"start": start, "end": end}
        if name:
            entry["name"] = name

        self._config.setdefault(mode, []).append(entry)
        self.name_edit.clear()
        self.start_offset_combo.setCurrentIndex(0)  # オフセットをリセット
        self.end_offset_combo.setCurrentIndex(0)  # オフセットをリセット
        self._refresh_mode_entries()

    def _on_delete_clicked(self):
        mode = self._current_mode()
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return

        rows = sorted(r.row() for r in sel)
        rows.reverse()  # 下から消す
        entries = self._config.get(mode, [])
        for r in rows:
            if 0 <= r < len(entries):
                entries.pop(r)
        self._refresh_mode_entries()

    def _on_cancel_clicked(self):
        # 何も保存せずに元の画面へ戻る
        sw = self.stacked_widget
        sw.setCurrentWidget(self.kpi_page)
        sw.removeWidget(self)
        self.deleteLater()

    def _on_save_clicked(self):
        # JSONファイルに保存
        try:
            _save_json(KPI_INTERVALS_PATH, self._config)
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"Failed to save kpi.json:\n{e}")
            return

        # 親ページに反映
        self.kpi_page._interval_config = self._config
        # KPI計算はエフォートごとに行うため、ここでは不要

        QMessageBox.information(self, "Saved Successfully", "kpi.json has been saved.")

        sw = self.stacked_widget
        sw.setCurrentWidget(self.kpi_page)
        sw.removeWidget(self)
        self.deleteLater()


# --------------------------------------------------------------------------------------
# KPI Page 本体
# --------------------------------------------------------------------------------------

class EffortRawDataPage(QDialog):
    """
    エフォートの生データを表示するダイアログ（別ウィンドウ）
    """
    
    def __init__(self, kpi_page: "KPIPage", effort_data: dict):
        super().__init__(kpi_page)
        self.kpi_page = kpi_page
        self.effort_data = effort_data
        
        # モーダルレス（非モーダル）で開くように設定
        self.setModal(False)
        # 閉じられたときに自動的に削除されるように設定
        self.setAttribute(Qt.WA_DeleteOnClose)
        # 常に前面に来ないようにウィンドウフラグを設定
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        
        self._build_ui()
    
    def _build_ui(self):
        self.setWindowTitle("Effort Raw Data")
        self.setMinimumSize(1000, 600)
        main_layout = QVBoxLayout()
        
        # タイトル
        title = QLabel(f"Effort Raw Data - {self.effort_data.get('player_name', 'Unknown')}")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        main_layout.addWidget(title)
        
        # エフォート情報
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel(f"Start: {self.effort_data.get('start_time')}"))
        info_layout.addWidget(QLabel(f"Date: {self.effort_data.get('date')}"))
        info_layout.addStretch()
        main_layout.addLayout(info_layout)
        
        # テーブル
        self.table = QTableView()
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setAlternatingRowColors(True)
        main_layout.addWidget(self.table)
        
        # ボタン
        button_layout = QHBoxLayout()
        
        # CSV保存ボタン
        save_csv_btn = QPushButton("Save as CSV")
        save_csv_btn.clicked.connect(self._save_to_csv)
        button_layout.addWidget(save_csv_btn)
        
        button_layout.addStretch()
        
        # 閉じるボタン
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
        # データを表示
        self._load_data()
    
    def _load_data(self):
        """エフォートの生データをテーブルに表示（TRACK_ORDERを列名としてtimestampを格納）"""
        if not self.effort_data.get("data_points"):
            self.model = DataFrameModel(pd.DataFrame())
            self.table.setModel(self.model)
            return
        
        # data_pointsをDataFrameに変換
        data_points = self.effort_data["data_points"]
        df = pd.DataFrame(data_points)
        
        if df.empty:
            self.model = DataFrameModel(pd.DataFrame())
            self.table.setModel(self.model)
            return
        
        # timestamp順にソート
        if "timestamp" not in df.columns or "position" not in df.columns:
            self.model = DataFrameModel(pd.DataFrame())
            self.table.setModel(self.model)
            return
        
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # TRACK_ORDERを列名として、各位置のtimestampを格納する形式に変換
        # 各周回ごとに1行を作成（0mの出現回数で周回を識別）
        zero_m_rows = df[df["position"] == "0m"].copy()
        
        if zero_m_rows.empty:
            # 0mがない場合は、全データを1周回として扱う
            rows_data = []
            row_dict = {}
            for pos in TRACK_ORDER:
                pos_rows = df[df["position"] == pos]
                if not pos_rows.empty:
                    row_dict[pos] = pos_rows.iloc[0]["timestamp"]
                else:
                    row_dict[pos] = None
            rows_data.append(row_dict)
            df_view = pd.DataFrame(rows_data)
        else:
            # 各0mの間を1周回として扱う
            rows_data = []
            
            for i in range(len(zero_m_rows)):
                # この周回の開始時刻（前の0m、またはエフォート開始時刻）
                if i == 0:
                    lap_start_time = df.iloc[0]["timestamp"]
                else:
                    lap_start_time = zero_m_rows.iloc[i-1]["timestamp"]
                
                # この周回の終了時刻（この0m）
                lap_end_time = zero_m_rows.iloc[i]["timestamp"]
                
                # この周回のデータを取得
                lap_df = df[
                    (df["timestamp"] >= lap_start_time) & 
                    (df["timestamp"] <= lap_end_time)
                ].copy()
                
                # 各位置のtimestampを抽出
                row_dict = {}
                for pos in TRACK_ORDER:
                    pos_rows = lap_df[lap_df["position"] == pos]
                    if not pos_rows.empty:
                        # 最初のtimestampを使用
                        row_dict[pos] = pos_rows.iloc[0]["timestamp"]
                    else:
                        row_dict[pos] = None
                
                rows_data.append(row_dict)
            
            df_view = pd.DataFrame(rows_data)
        
        # 列の順序をTRACK_ORDERに合わせる
        df_view = df_view[TRACK_ORDER]
        
        self.model = DataFrameModel(df_view)
        self.table.setModel(self.model)
        
        # 見た目調整
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Interactive)
        
        default_width = 160
        visible_cols = list(self.model._df.columns)
        for i, c in enumerate(visible_cols):
            self.table.setColumnWidth(i, default_width)
        self.table.verticalHeader().setDefaultSectionSize(26)
        
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
    
    def _save_to_csv(self):
        """エフォートの生データをCSVファイルに保存"""
        if not hasattr(self, 'model') or self.model._df.empty:
            QMessageBox.warning(self, "Error", "No data to save")
            return
        
        # ファイル保存ダイアログ
        player_name = self.effort_data.get('player_name', 'Unknown').replace(' ', '_')
        date_str = ""
        if self.effort_data.get('date'):
            date = self.effort_data['date']
            if isinstance(date, (pd.Timestamp, datetime.datetime)):
                date_str = date.strftime("%Y%m%d_%H%M%S")
            else:
                date_str = str(date).replace(' ', '_').replace(':', '')
        
        default_filename = f"effort_{player_name}_{date_str}.csv"
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV",
            default_filename,
            "CSV Files (*.csv)"
        )
        
        if not path:
            return
        
        try:
            # DataFrameをCSVに保存
            self.model._df.to_csv(path, index=False, encoding="utf-8-sig")
            QMessageBox.information(self, "Save Complete", f"CSV file saved:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Failed to save CSV file:\n{e}")
    
class KPIPage(QWidget):
    """
    KPIページ
    - データ処理ロジック（DB 取得）は _load_data 系に集約
    - UI 構築は _build_ui で担当
    """

    # ------------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------------
    def __init__(self, query: str, stacked_widget, user_ids):
        super().__init__()

        self._base_query = query
        self.stacked_widget = stacked_widget
        self._user_ids = list(map(int, user_ids))

        self._settings = _load_json(SETTINGS_PATH)
        # kpi.jsonを読み込む
        self._interval_config = _load_json(KPI_INTERVALS_PATH) or {}
        
        # versionチェック
        config_version = self._interval_config.get("version")
        if config_version != "ver2":
            version_msg = f"Found: {config_version}" if config_version else "Version key not found"
            QMessageBox.critical(
                self,
                "Version Error",
                f"Invalid kpi.json version.\nExpected: ver2\n{version_msg}\n\nPlease update kpi.json to version ver2."
            )
            # アプリを終了
            import sys
            sys.exit(1)
        


        # time_mode 初期値
        ui_mode = (self._settings.get("ui", {}).get("time_mode") or "rolling").lower()
        if ui_mode in ("rs", "rolling"):
            self.time_mode = "rolling"
        elif ui_mode in ("standing",):
            self.time_mode = "standing"
        elif ui_mode in ("fly", "flying"):
            self.time_mode = "flying"
        else:
            self.time_mode = "rolling"

        # データ読み込み（Data ロジック）
        self.df_all = pd.DataFrame()
        # CSVファイルパスの場合は_load_data_by_csv、それ以外は_load_data
        if isinstance(self._base_query, str) and (self._base_query.endswith('.csv') or self._base_query.startswith('CSV_FILE:')):
            csv_path = self._base_query.replace('CSV_FILE:', '') if self._base_query.startswith('CSV_FILE:') else self._base_query
            self._load_data_by_csv(csv_path)
        else:
            self._load_data()

        # UI 構築（UI ロジック）
        self._build_ui()
        
        # エフォート検出とテーブル更新
        self._detect_and_display_efforts()
        
        self.track_image_path = self._settings.get("image_path", "")
    
    def mousePressEvent(self, event):
        """マウスクリック時にウィンドウを前面に表示"""
        super().mousePressEvent(event)
        # 親ウィンドウを取得して前面に表示
        parent_window = self.window()
        if parent_window:
            parent_window.raise_()
            parent_window.activateWindow()
 
    # ------------------------------------------------------------------
    # データロジック
    # ------------------------------------------------------------------
    def _load_data(self):
        """DB から生データを取得する（シンプル版）"""
        df = get_df_from_db(self._base_query)

        if df.empty:
            QMessageBox.warning(
                self,
                "No Data",
                "No data found for the selected conditions.\nReturning to the previous page."
            )
            self.go_back()
            return
        
        # タイムスタンプをJST（naive）へ変換
        if "timestamp" in df.columns:
            df["timestamp"] = to_jst_naive(df["timestamp"])
        
        # デコーダ→地点名
        if "decoder_id" in df.columns:
            df["position"] = df["decoder_id"].map(translate_dict).fillna("Unknown")
        else:
            df["position"] = "Unknown"
        
        # 時系列でソート
        self.df_all = df.sort_values("timestamp").reset_index(drop=True)
        
        print(f"[データ読み込み] 完了: {len(self.df_all)}行")
    
    def _load_data_by_csv(self, csv_path: str):
        """CSVファイルから生データを取得する"""
        print(f"[CSV読み込み] ファイル: {csv_path}")
        
        # 複数のエンコーディングを試す
        encodings = ["utf-8-sig", "utf-8", "shift_jis", "cp932", "euc-jp"]
        df = None
        last_error = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                break
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue
        
        if df is None:
            error_msg = f"CSVファイルの読み込みに失敗しました: {csv_path}\nすべてのエンコーディングで読み込みに失敗しました。"
            if last_error:
                error_msg += f"\n最後のエラー: {last_error}"
            QMessageBox.warning(
                self,
                "CSV読み込みエラー",
                error_msg
            )
            self.go_back()
            return
        
        if df.empty:
            QMessageBox.warning(
                self,
                "No Data",
                "CSVファイルにデータが含まれていません。\nReturning to the previous page."
            )
            self.go_back()
            return
        
        # タイムスタンプをJST（naive）へ変換
        # CSVの場合は既にJSTの可能性があるので、datetime型に変換してから処理
        if "timestamp" in df.columns:
            # 文字列の場合はdatetimeに変換
            if df["timestamp"].dtype == 'object':
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
            else:
                # 既にdatetime型の場合も確実にdatetime型にする
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
            
            # タイムゾーン情報がある場合はJSTに変換
            # pandasのDatetimeTZDtypeをチェック
            if hasattr(df["timestamp"].dtype, 'tz') and df["timestamp"].dtype.tz is not None:
                df["timestamp"] = to_jst_naive(df["timestamp"])
            # タイムゾーン情報がない場合はそのまま（既にJSTと仮定）
        
        # デコーダ→地点名
        if "decoder_id" in df.columns:
            df["position"] = df["decoder_id"].map(translate_dict).fillna("Unknown")
        elif "position" not in df.columns:
            # position列がない場合はUnknownを設定
            df["position"] = "Unknown"
        
        # 時系列でソート
        self.df_all = df.sort_values("timestamp").reset_index(drop=True)
        
        print(f"[CSV読み込み] 完了: {len(self.df_all)}行")
        

    def _display_kpi_columns(self) -> list[str]:
        """
        現在のモード(self.time_mode)について、
        kpi.json の start/end 定義から生成される KPI 列名を返す。

        - name があれば name
        - なければ "start-end"
        """
        mode = (self.time_mode or "rolling").lower()
        cfg = self._interval_config or {}
        entries = cfg.get(mode, [])

        cols: list[str] = []
        for ent_idx, ent in enumerate(entries):
            if not isinstance(ent, dict):
                continue
            start = ent.get("start")
            end = ent.get("end")
            name = ent.get("name")
            
            if not start or not end:
                continue
            
            # nameがあればname、なければ "start-end" を使用
            col_name = name if name else f"{start}-{end}"
            cols.append(col_name)

        return cols

    # ------------------------------------------------------------------
    # エフォート検出ロジック
    # ------------------------------------------------------------------
    def _calculate_kpi_for_effort(self, effort_df: pd.DataFrame) -> pd.DataFrame:
        """
        エフォートのDataFrameに対してKPI列を計算する
        
        Args:
            effort_df: エフォートのデータポイントから作成したDataFrame（各行が1つの位置のデータポイント）
            
        Returns:
            KPI列が追加されたDataFrame
        """
        if effort_df.empty:
            return effort_df
        
        # 結果DataFrameをコピー
        result_df = effort_df.copy()
        
        # timestamp列とposition列が必要
        if "timestamp" not in result_df.columns or "position" not in result_df.columns:
            return result_df
        
        # timestamp順にソート
        result_df = result_df.sort_values("timestamp").reset_index(drop=True)
        
        # 現在のモードに基づいてKPI列を計算
        cfg = self._interval_config or {}
        mode = (self.time_mode or "rolling").lower()
        entries = cfg.get(mode, [])
        
        if not isinstance(entries, list):
            return result_df
        
        def parse_position_with_offset(pos_str):
            """地点名から位置名とオフセットを抽出（例: "0m+1" -> ("0m", 1)）"""
            if not pos_str:
                return None, 0
            match = re.match(r'^(.+?)(\+(\d+))?$', pos_str)
            if match:
                pos_name = match.group(1)
                offset_str = match.group(3)
                offset = int(offset_str) if offset_str else 0
                return pos_name, offset
            return pos_str, 0
        
        def find_position_timestamp(position_name, offset=0):
            """指定された位置のtimestampを取得（offsetでn個後の位置を指定）"""
            matching_rows = result_df[result_df["position"] == position_name]
            if matching_rows.empty:
                return None
            
            idx = offset
            if idx < len(matching_rows):
                timestamp = matching_rows.iloc[idx]["timestamp"]
                return timestamp
            return None
        
        for ent_idx, ent in enumerate(entries):
            if not isinstance(ent, dict):
                continue
                
            start_str = ent.get("start")
            end_str = ent.get("end")
            name_str = ent.get("name")
            
            if not start_str or not end_str:
                continue
            
            # 位置名とオフセットを解析
            start_pos, start_offset = parse_position_with_offset(start_str)
            end_pos, end_offset = parse_position_with_offset(end_str)
            
            # 列名はnameを使用（nameがなければ "start-end"）
            col_name = name_str if name_str else f"{start_str}-{end_str}"
            
            # すでに列があるなら再計算しない
            if col_name in result_df.columns:
                continue
            
            # 開始位置と終了位置のtimestampを取得
            start_time = find_position_timestamp(start_pos, start_offset)
            end_time = find_position_timestamp(end_pos, end_offset)
            
            # 該当するデータがなければNaN
            if start_time is None or end_time is None:
                result_df[col_name] = math.nan
                continue
            
            # 時刻の差を計算（秒）
            try:
                time_diff = (end_time - start_time).total_seconds()
                kpi_value = round(time_diff, 3)
                # すべての行に同じ値を設定
                result_df[col_name] = kpi_value
            except Exception as e:
                print(f"[KPI計算エラー] {col_name}: {e}")
                result_df[col_name] = math.nan
        
        return result_df
    
    def _detect_and_display_efforts(self):
        """
        エフォートを検出してテーブルに表示する。
        
        ルール:
        - 0mを検出（起点0m）
        - 起点0mから5秒前の区間にSB1があればstartとしてその時刻を、SB1がなくFPがあればその時刻をstartに設定。どちらもなければエフォートIDを採番しない
        - 起点0m以降のFPをすべて取得。FPの間隔が30秒以上開く箇所を探し、その30秒以上開いたFPまでを同一エフォートデータとして保持
        - これを0m毎に繰り返す
        """
        efforts = []  # 最初に初期化
                
        # 必要な列の存在確認
        if "user_id" not in self.df_all.columns or "position" not in self.df_all.columns or "timestamp" not in self.df_all.columns:
            self.effort_table.setRowCount(0)
            self._efforts_data = []
            return
        
        # 選手名の列を確認
        has_first_name = "first_name" in self.df_all.columns
        has_last_name = "last_name" in self.df_all.columns
        
        # Date列を確認
        has_date = "Date" in self.df_all.columns
        
        # user_idが空欄のSB1データを取得（全選手で共有）
        sb1_no_user = self.df_all[
            (self.df_all["position"] == "SB1") & 
            (self.df_all["user_id"].isna() | (self.df_all["user_id"] == ""))
        ].copy()
        
        # 選手ごとにグループ化
        for user_id, group in self.df_all.groupby("user_id"):
            # user_idが空欄のグループはスキップ（SB1は後でマージする）
            if pd.isna(user_id) or user_id == "":
                continue
            
            # 選手名を取得
            player_name = "Unknown"
            if has_first_name or has_last_name:
                first_row = group.iloc[0]
                name_parts = []
                if has_first_name and pd.notna(first_row.get("first_name")):
                    name_parts.append(str(first_row["first_name"]))
                if has_last_name and pd.notna(first_row.get("last_name")):
                    name_parts.append(str(first_row["last_name"]))
                if name_parts:
                    player_name = " ".join(name_parts)
            
            
            # この選手のデータとuser_idが空欄のSB1データをマージ
            group_with_sb1 = pd.concat([group, sb1_no_user], ignore_index=True)
            
            # 時系列でソート
            group_sorted_all = group_with_sb1.sort_values("timestamp").reset_index(drop=True)
            
            # 0mの位置を持つ行を検出
            zero_m_rows = group_sorted_all[group_sorted_all["position"] == "0m"].copy()
            
            if zero_m_rows.empty:
                continue
            
            # 各0mについて独立にエフォートを検出
            for idx in range(len(zero_m_rows)):
                zero_m_row = zero_m_rows.iloc[idx]
                zero_m_time = zero_m_row["timestamp"]  # 起点0m
                
                if pd.isna(zero_m_time):
                    continue
                
                # 起点0mから5秒前の区間にSB1またはFPがあるかチェック
                start_time = None
                start_type = None
                
                # 全データから、起点0mの5秒前から起点0mまでの範囲でSB1またはFPを探す
                check_start_time = zero_m_time - pd.Timedelta(seconds=5)
                
                # 検索範囲内の行を取得
                mask = (group_sorted_all["timestamp"] >= check_start_time) & (group_sorted_all["timestamp"] <= zero_m_time)
                search_rows = group_sorted_all[mask]
                
                # SB1を優先して検索
                sb1_rows = search_rows[search_rows["position"] == "SB1"]
                if not sb1_rows.empty:
                    start_time = sb1_rows.iloc[-1]["timestamp"]  # 最後のSB1（0mに最も近い）
                    start_type = "SB1"
                else:
                    # FPを検索
                    fp_rows = search_rows[search_rows["position"] == "FP"]
                    if not fp_rows.empty:
                        start_time = fp_rows.iloc[-1]["timestamp"]  # 最後のFP（0mに最も近い）
                        start_type = "FP"
                
                # startが設定されていない場合はエフォートIDを採番しない
                if start_time is None:
                    continue
                
                # startから30秒以内のFPを順に追跡
                # 見つからなくなるまで繰り返し、最後のFPから30秒後までのデータをエフォートとして確定
                current_time = start_time
                last_fp_time = None
                
                # 起点0m以降のすべてのFPを時系列で取得
                fp_rows_after = group_sorted_all[
                    (group_sorted_all["timestamp"] > zero_m_time) & 
                    (group_sorted_all["position"] == "FP")
                ]
                fps_after_zero_m = fp_rows_after["timestamp"].tolist()
                
                # startから30秒以内のFPを順に追跡
                while True:
                    # current_timeから30秒以内のFPを探す
                    found_fp = None
                    for fp_time in fps_after_zero_m:
                        time_diff = (fp_time - current_time).total_seconds()
                        if 0 < time_diff <= 30:
                            found_fp = fp_time
                            break
                    
                    if found_fp is None:
                        # 30秒以内のFPが見つからなかった
                        if last_fp_time is None:
                            # FPが1つも見つからなかった場合
                            # startから30秒後までのデータを含める
                            end_time = start_time + pd.Timedelta(seconds=30)
                            
                            # startから30秒後までの間にあるすべてのデータポイントを取得
                            mask = (group_sorted_all["timestamp"] >= start_time) & (group_sorted_all["timestamp"] <= end_time)
                            effort_data = group_sorted_all[mask].sort_values("timestamp")
                            
                            # 辞書形式に変換
                            effort_data_points = effort_data.to_dict("records")
                            
                            start_date = zero_m_row.get("Date") if has_date else start_time
                            efforts.append({
                                "player_name": player_name,
                                "date": start_date,
                                "start_time": start_time,
                                "start_type": start_type,
                                "data_points": effort_data_points
                            })
                            break
                        else:
                            # 最後のFPから30秒後までのデータをエフォートとして確定
                            end_time = last_fp_time + pd.Timedelta(seconds=30)
                            
                            # startからend_timeまでの間にあるすべてのデータポイントを取得
                            mask = (group_sorted_all["timestamp"] >= start_time) & (group_sorted_all["timestamp"] <= end_time)
                            effort_data = group_sorted_all[mask].sort_values("timestamp")
                            
                            # 辞書形式に変換
                            effort_data_points = effort_data.to_dict("records")
                            
                            # エフォートを確定
                            start_date = zero_m_row.get("Date") if has_date else start_time
                            efforts.append({
                                "player_name": player_name,
                                "date": start_date,
                                "start_time": start_time,
                                "start_type": start_type,
                                "data_points": effort_data_points
                            })
                            break
                    else:
                        # 見つかったFPを次のcurrent_timeとして設定
                        last_fp_time = found_fp
                        current_time = found_fp
        
        # エフォートごとにKPIを計算
        print(f"[エフォート検出] 合計 {len(efforts)} 個のエフォートを検出")
        
        for effort_idx, effort in enumerate(efforts):
            if not effort.get("data_points"):
                continue
            
            # data_pointsをDataFrameに変換
            effort_df = pd.DataFrame(effort["data_points"])
            
            if effort_df.empty:
                continue
            
            # KPI列を計算
            effort_df_with_kpi = self._calculate_kpi_for_effort(effort_df)
            
            # 計算したKPI列を含むdata_pointsに更新
            effort["data_points"] = effort_df_with_kpi.to_dict("records")
        
        # エフォートデータを保持
        self._efforts_data = efforts.copy() if efforts else []
        
        # エフォートテーブルを更新
        self._update_effort_table_display()
    
    def _update_effort_table_display(self):
        """エフォートテーブルの表示を更新（モード変更時にも呼び出される）"""
        if not hasattr(self, '_efforts_data') or not self._efforts_data:
            self.effort_table.setRowCount(0)
            return
        
        efforts = self._efforts_data
        
        # KPI列のリストを取得（現在のモードに基づく）
        kpi_cols = self._display_kpi_columns()
        
        # エフォートテーブルの列数を設定: 選手名、日時、[KPI列...]、周回数
        base_cols_before_kpi = 2  # 選手名、日時
        base_cols_after_kpi = 1  # 周回数
        total_cols = base_cols_before_kpi + len(kpi_cols) + base_cols_after_kpi
        self.effort_table.setColumnCount(total_cols)
        
        # ヘッダーラベルを設定
        headers = ["PlayerName", "Date"] + kpi_cols + ["LapCount"]
        self.effort_table.setHorizontalHeaderLabels(headers)
        
        # ヘッダーのリサイズモードを設定（下のテーブルと同じ仕様）
        hh = self.effort_table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Interactive)
        
        # 列幅を設定
        default_width = 160
        wider = {"PlayerName":220, "Date": 220, "LapCount": 80}
        for i, header in enumerate(headers):
            self.effort_table.setColumnWidth(i, wider.get(header, default_width))
        
        # エフォートテーブルを更新
        self.effort_table.setRowCount(len(efforts))
        for idx, effort in enumerate(efforts):
            # 選手名
            self.effort_table.setItem(idx, 0, QTableWidgetItem(str(effort["player_name"])))
            
            # 日時
            date_str = ""
            if effort["date"] is not None:
                if isinstance(effort["date"], (pd.Timestamp, datetime.datetime)):
                    date_str = effort["date"].strftime("%Y-%m-%d %H:%M:%S")
                else:
                    date_str = str(effort["date"])
            self.effort_table.setItem(idx, 1, QTableWidgetItem(date_str))
            
            # KPI列の値を表示
            if effort.get("data_points") and len(effort["data_points"]) > 0:
                effort_df = pd.DataFrame(effort["data_points"])
                
                for kpi_idx, kpi_col in enumerate(kpi_cols):
                    col_idx = base_cols_before_kpi + kpi_idx
                    
                    if kpi_col in effort_df.columns:
                        # KPI列の値を取得（エフォート全体で計算された値）
                        kpi_values = pd.to_numeric(effort_df[kpi_col], errors='coerce')
                        # NaN以外の値を取得（通常は1つの値のはず）
                        valid_values = kpi_values.dropna()
                        
                        if len(valid_values) > 0:
                            # 最初の有効な値を使用（通常は1つだけ）
                            kpi_value = valid_values.iloc[0]
                            # 小数点以下3桁で表示
                            kpi_str = f"{kpi_value:.3f}"
                        else:
                            kpi_str = ""
                        
                        self.effort_table.setItem(idx, col_idx, QTableWidgetItem(kpi_str))
                    else:
                        self.effort_table.setItem(idx, col_idx, QTableWidgetItem(""))
            
            # 周回数（エフォート内の0mの回数をカウント）
            lap_count = 0
            if effort.get("data_points"):
                # 生データではposition列で0mを判定
                for point in effort["data_points"]:
                    if point.get("position") == "0m":
                        lap_count += 1
            
            lap_col_idx = base_cols_before_kpi + len(kpi_cols)
            self.effort_table.setItem(idx, lap_col_idx, QTableWidgetItem(str(lap_count)))

    # ------------------------------------------------------------------
    # UIロジック
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.setWindowTitle("LapApp Ver2 - Display KPIs")
        self.resize(980, 640)

        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel("KPIs"))

        # --- 上部バー（Show All, モード切替, Reload） ---
        top_row = QHBoxLayout()


        self.mode_group = QButtonGroup(self)
        self.rb_roll = QRadioButton("Rolling")
        self.rb_stand = QRadioButton("Standing")
        self.rb_fly = QRadioButton("Flying")
        self.mode_group.addButton(self.rb_roll)
        self.mode_group.addButton(self.rb_stand)
        self.mode_group.addButton(self.rb_fly)

        self.rb_roll.setChecked(self.time_mode == "rolling")
        self.rb_stand.setChecked(self.time_mode == "standing")
        self.rb_fly.setChecked(self.time_mode == "flying")

        top_row.addWidget(self.rb_roll)
        top_row.addWidget(self.rb_stand)
        top_row.addWidget(self.rb_fly)

        self.btnReload = QPushButton("Updata to Latest", self)
        self.btnReload.clicked.connect(self._reload_kpi)
        top_row.addWidget(self.btnReload)
        
        self.btnEditKpiJson = QPushButton("Edit KPI Setting", self)
        self.btnEditKpiJson.clicked.connect(self._open_kpi_json_editor)
        top_row.addWidget(self.btnEditKpiJson)

        # ショートカット: Ctrl+R でリロード
        reload_action = QAction(self)
        reload_action.setShortcut("Ctrl+R")
        reload_action.triggered.connect(self._reload_kpi)
        self.addAction(reload_action)

        main_layout.addLayout(top_row)


        # --- エフォートテーブル ---
        effort_label = QLabel("Effort List")
        main_layout.addWidget(effort_label)
        
        self.effort_table = QTableWidget(0, 5)
        self.effort_table.setHorizontalHeaderLabels(["Player", "Date", "start", "end", "the number of laps"])

        hh = self.effort_table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Interactive)
        self.effort_table.setAlternatingRowColors(True)
        self.effort_table.verticalHeader().setVisible(True)
        self.effort_table.verticalHeader().setDefaultSectionSize(26)
        self.effort_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.effort_table.setSelectionMode(QAbstractItemView.SingleSelection)
        main_layout.addWidget(self.effort_table)
        
        # 生データを確認ボタン
        btn_view_raw_data = QPushButton("View Raw Data for Selected Effort")
        btn_view_raw_data.clicked.connect(self._view_effort_raw_data)
        main_layout.addWidget(btn_view_raw_data)

        # 戻るボタン
        back_btn = QPushButton("← Back to Main")
        back_btn.clicked.connect(self.go_back)
        main_layout.addWidget(back_btn)

        self.setLayout(main_layout)

        # シグナル接続（UI→挙動）
        self.rb_roll.toggled.connect(
            lambda checked: checked and self._on_mode_changed("rolling")
        )
        self.rb_stand.toggled.connect(
            lambda checked: checked and self._on_mode_changed("standing")
        )
        self.rb_fly.toggled.connect(
            lambda checked: checked and self._on_mode_changed("flying")
        )

    # ---- モード変更 ----
    def _on_mode_changed(self, mode: str):
        if mode == getattr(self, "time_mode", None):
            return
        self.time_mode = mode
        self._settings.setdefault("ui", {})["time_mode"] = mode
        _save_json(SETTINGS_PATH, self._settings)

        # モード変更時はKPIを再計算してからテーブルを更新
        if hasattr(self, '_efforts_data') and self._efforts_data:
            for effort_idx, effort in enumerate(self._efforts_data):
                if not effort.get("data_points"):
                    continue
                
                # data_pointsをDataFrameに変換
                effort_df = pd.DataFrame(effort["data_points"])
                
                if effort_df.empty:
                    continue
                
                # KPI列を再計算（新しいモードに応じたKPI列が計算される）
                effort_df_with_kpi = self._calculate_kpi_for_effort(effort_df)
                
                # 計算したKPI列を含むdata_pointsに更新
                effort["data_points"] = effort_df_with_kpi.to_dict("records")

        # エフォートテーブルも更新（KPI列が変わるため）
        self._update_effort_table_display()

    # ---- 戻る ----
    def go_back(self):
        sw = self.stacked_widget
        idx = sw.indexOf(self)
        if idx != -1:
            sw.removeWidget(self)
        sw.setCurrentIndex(0)

    # ---- リロード ----
    def _reload_kpi(self):
        # データを再読込
        self._load_data()

        # エフォート検出とテーブル更新
        self._detect_and_display_efforts()

    def _reload_kpi_json(self):
        # KPI計算はエフォートごとに行うため、ここでは不要
        pass

    def _open_kpi_json_editor(self):
        """kpi.json編集ページへ遷移"""
        editor = KPIJsonEditorPage(self)
        self.stacked_widget.addWidget(editor)
        self.stacked_widget.setCurrentWidget(editor)
    
    def _view_effort_raw_data(self):
        """選択されたエフォートの生データを表示"""
        selected_rows = self.effort_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "選択エラー", "エフォートを選択してください")
            return
        
        row_idx = selected_rows[0].row()
        
        if not hasattr(self, '_efforts_data') or not self._efforts_data:
            QMessageBox.warning(self, "エラー", "エフォートデータが読み込まれていません")
            return
        
        if row_idx < 0 or row_idx >= len(self._efforts_data):
            QMessageBox.warning(
                self, 
                "エラー", 
                f"無効な行が選択されています\n行: {row_idx}\nデータ数: {len(self._efforts_data)}"
            )
            return
        
        effort_data = self._efforts_data[row_idx]
        
        # エフォート生データダイアログを別ウィンドウで開く（モーダルレス）
        raw_data_dialog = EffortRawDataPage(self, effort_data)
        raw_data_dialog.show()
