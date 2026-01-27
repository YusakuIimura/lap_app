from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QHBoxLayout, QTableView, QFormLayout, QCheckBox,
    QAbstractItemView, QMessageBox, QFileDialog, QRadioButton, QButtonGroup, QHeaderView,
    QMenu, QAction, QComboBox,QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, QPoint
from PyQt5.QtGui import QBrush, QColor, QGuiApplication, QKeySequence, QPixmap

import math
import datetime
import numpy as np
import pandas as pd
import json
import os

from utils import fetch_df_from_db

# --------------------------------------------------------------------------------------
# 共通定数 / ヘルパ
# --------------------------------------------------------------------------------------

SETTINGS_PATH = os.path.join(os.getcwd(), "settings.json")
KPI_INTERVALS_PATH = os.path.join(os.getcwd(), "kpi.json")

TRACK_ORDER = [
    "FP_start",
    "SB1",
    "0m_start",
    "60m",
    "AP1",
    "50m",
    "100m",
    "BP",
    "150m",
    "AP2",
    "200m",
    "FP_2nd",
    "0m_2nd",
]
# 名前系は常に先頭に出したい列
NAME_COLUMNS = ["first_name", "last_name", "Date", "FP_start"]


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

def _load_kpi_intervals(path: str) -> dict:
    """kpi.json から interval 定義を読み込む（壊れていたら空 dict）"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


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

        # start/end で選べる地点（df_all にある列だけを採用）
        self.available_points = [
            p for p in TRACK_ORDER if p in kpi_page.df_all.columns
        ]

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
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Display Name（start-end if empty）")

        add_row.addWidget(QLabel("start:"))
        add_row.addWidget(self.start_combo)
        add_row.addWidget(QLabel("end:"))
        add_row.addWidget(self.end_combo)
        add_row.addWidget(QLabel("Display Name:"))
        add_row.addWidget(self.name_edit)

        self.btnAdd = QPushButton("Add")
        self.btnAdd.clicked.connect(self._on_add_clicked)
        add_row.addWidget(self.btnAdd)

        main.addLayout(add_row)

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
        start = self.start_combo.currentText()
        end = self.end_combo.currentText()
        name = self.name_edit.text().strip()

        if not start or not end:
            QMessageBox.warning(self, "Cannot add.", "Please select both start and end.")
            return

        entry = {"start": start, "end": end}
        if name:
            entry["name"] = name

        self._config.setdefault(mode, []).append(entry)
        self.name_edit.clear()
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

        # 親ページに反映してテーブル更新
        self.kpi_page._interval_config = self._config
        self.kpi_page._ensure_interval_columns()
        self.kpi_page._build_filters()
        self.kpi_page._rebuild_table(self.kpi_page.debug_all_cols.isChecked())

        QMessageBox.information(self, "Saved Successfully", "kpi.json has been saved.")

        sw = self.stacked_widget
        sw.setCurrentWidget(self.kpi_page)
        sw.removeWidget(self)
        self.deleteLater()


# --------------------------------------------------------------------------------------
# KPI Page 本体
# --------------------------------------------------------------------------------------

class KPIPage(QWidget):
    """
    KPIページ
    - データ処理ロジック（DB 取得 / KPI 計算）は _load_data_and_prepare_kpi 系に集約
    - UI 構築は _build_ui / _build_filters / _rebuild_table で担当
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
        self._interval_config: dict = {}

        # フィルタ用
        self.filter_kpi_combo: QComboBox | None = None
        self.filter_min_edit: QLineEdit | None = None
        self.filter_max_edit: QLineEdit | None = None
        self._filter_stats: dict[str, tuple[float | None, float | None]] = {}
        self._current_filter_col: str | None = None

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

        # データ読み込み + KPI 準備（Data ロジック）
        self.df_all = pd.DataFrame()
        self._load_data_and_prepare_kpi()

        # UI 構築（UI ロジック）
        self._build_ui()

        # フィルタ・テーブル初期化
        self._build_filters()
        self._rebuild_table(self.debug_all_cols.isChecked())
        
        self.track_image_path = self._settings.get("image_path", "")
 
    # ------------------------------------------------------------------
    # データロジック
    # ------------------------------------------------------------------
    def _load_data_and_prepare_kpi(self):
        """DB から df_all を取得し、区間列とモード別 KPI を構築する。"""
        df_any, _users = fetch_df_from_db(self._base_query, progress=lambda m: print(f"[KPI] {m}"))
        df_any = df_any.copy()

        if "__row_id" not in df_any.columns:
            df_any["__row_id"] = range(len(df_any))

        self.df_all = df_any

        # kpi.json から interval 定義を読み込み
        self._interval_config = _load_kpi_intervals(KPI_INTERVALS_PATH)

        # ラップタイム列から区間タイム列を追加
        self._ensure_interval_columns()

    def _ensure_interval_columns(self):
        """
        kpi.json の start/end 定義に基づき、区間タイム列を self.df_all に追加する。

        - name があれば列名に使用
        - name がなければ "start-end" を列名にする
        - TRACK_ORDER 上で start が end より後ろの場合は
          「start(この周回) → end(次周)」として計算する
        """
        if self.df_all.empty:
            return

        cfg = self._interval_config or {}
        pos_index = {name: i for i, name in enumerate(TRACK_ORDER)}

        for mode, entries in cfg.items():
            if not isinstance(entries, list):
                continue

            for ent in entries:
                start = ent.get("start")
                end = ent.get("end")
                if not start or not end:
                    continue

                col_name = ent.get("name") or f"{start}-{end}"

                # すでに列があるなら再計算しない
                if col_name in self.df_all.columns:
                    continue

                # 必要な地点タイムが無ければスキップ
                if start not in self.df_all.columns or end not in self.df_all.columns:
                    continue

                s = self.df_all[start]
                e = self.df_all[end]

                # 向きの判定：TRACK_ORDER に両方ある場合だけ厳密な比較
                idx_s = pos_index.get(start)
                idx_e = pos_index.get(end)

                if idx_s is not None and idx_e is not None and idx_s > idx_e:
                    # start の方が"後" → end は次周の値を使う
                    e_series = e.shift(-1)
                else:
                    # 通常: 同一周回内
                    e_series = e

                # None や NaN を含む行は NaN として処理
                mask_valid = pd.notna(s) & pd.notna(e_series)
                diff = pd.Series(index=s.index, dtype='timedelta64[ns]')
                diff[mask_valid] = e_series[mask_valid] - s[mask_valid]
                diff[~mask_valid] = pd.NaT

                # datetime → Timedelta → 秒
                try:
                    result = pd.Series(index=diff.index, dtype=float)
                    valid_mask = pd.notna(diff)
                    result[valid_mask] = diff[valid_mask].dt.total_seconds().round(3)
                    result[~valid_mask] = math.nan
                    self.df_all[col_name] = result
                except Exception:
                    # 念のためのフォールバック
                    def _calc(row):
                        t0 = row.get(start)
                        t1 = row.get(end)
                        if idx_s is not None and idx_e is not None and idx_s > idx_e:
                            # 次周の end
                            # row 単位では next row を取れないので NaN 扱い
                            return math.nan
                        if pd.isna(t0) or pd.isna(t1):
                            return math.nan
                        try:
                            return float((t1 - t0).total_seconds())
                        except Exception:
                            return math.nan

                    self.df_all[col_name] = self.df_all.apply(_calc, axis=1)


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
        for ent in entries:
            if not isinstance(ent, dict):
                continue
            start = ent.get("start")
            end = ent.get("end")
            if not start or not end:
                continue
            name = ent.get("name") or f"{start}-{end}"
            cols.append(name)

        # 実際に DataFrame に存在するものだけ
        return [c for c in cols if c in self.df_all.columns]

    # ------------------------------------------------------------------
    # UIロジック
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.setWindowTitle("Display KPIs")
        self.resize(980, 640)

        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel("KPIs"))

        # --- 上部バー（Show All, モード切替, Reload） ---
        top_row = QHBoxLayout()

        self.debug_all_cols = QCheckBox("Show All Columns")
        self.debug_all_cols.setChecked(False)
        top_row.addWidget(self.debug_all_cols)

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

        # --- フィルタフォーム（1行だけ：KPI選択 + min/max） ---
        self.filter_form = QFormLayout()
        main_layout.addLayout(self.filter_form)

        # --- テーブル ---
        self.table = QTableView()
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setAlternatingRowColors(True)

        # 右クリックメニュー
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context_menu)

        # Ctrl/Cmd + C でコピー
        copy_shortcut = QAction(self.table)
        copy_shortcut.setShortcut(QKeySequence.Copy)
        copy_shortcut.triggered.connect(self._copy_selected_rows_to_clipboard)
        self.table.addAction(copy_shortcut)

        main_layout.addWidget(self.table)

        # --- 下部ボタン群 ---
        btn_row_delete = QPushButton("Delete selected rows")
        btn_export_csv = QPushButton("Export to CSV")
        btn_row_delete.clicked.connect(self.delete_selected_rows)
        btn_export_csv.clicked.connect(self.export_current_view_to_csv)

        btn_box = QHBoxLayout()
        btn_box.addWidget(btn_row_delete)
        btn_box.addWidget(btn_export_csv)
        main_layout.addLayout(btn_box)

        # 戻るボタン
        back_btn = QPushButton("← Back to Main")
        back_btn.clicked.connect(self.go_back)
        main_layout.addWidget(back_btn)

        self.setLayout(main_layout)

        # シグナル接続（UI→挙動）
        self.debug_all_cols.stateChanged.connect(
            lambda _s: self._rebuild_table(self.debug_all_cols.isChecked())
        )
        self.rb_roll.toggled.connect(
            lambda checked: checked and self._on_mode_changed("rolling")
        )
        self.rb_stand.toggled.connect(
            lambda checked: checked and self._on_mode_changed("standing")
        )
        self.rb_fly.toggled.connect(
            lambda checked: checked and self._on_mode_changed("flying")
        )

    # ---- フィルタ UI ----
    def _build_filters(self):
        # 既存行をクリア
        while self.filter_form.rowCount():
            self.filter_form.removeRow(0)

        self.filter_kpi_combo = None
        self.filter_min_edit = None
        self.filter_max_edit = None
        self._filter_stats = {}
        self._current_filter_col = None

        numeric_cols = self._display_kpi_columns()
        if not numeric_cols or self.df_all.empty:
            label = QLabel("KPI Filter: 利用可能な数値KPIがありません")
            self.filter_form.addRow(label)
            return

        # Q1, Q3 を計算
        num_df = self.df_all[numeric_cols].apply(pd.to_numeric, errors="coerce")
        q1 = num_df.quantile(0.25, numeric_only=True)
        q3 = num_df.quantile(0.75, numeric_only=True)
        for col in numeric_cols:
            self._filter_stats[col] = (
                float(q1[col]) if pd.notna(q1[col]) else None,
                float(q3[col]) if pd.notna(q3[col]) else None,
            )

        row = QHBoxLayout()
        combo = QComboBox()
        combo.addItem("（No Filter）", userData=None)
        for col in numeric_cols:
            combo.addItem(col, userData=col)

        min_edit = QLineEdit()
        min_edit.setPlaceholderText("min（Fallback to first quartile if empty）")

        max_edit = QLineEdit()
        max_edit.setPlaceholderText("max（Fallback to third quartile if empty）")

        row.addWidget(combo)
        row.addWidget(min_edit)
        row.addWidget(QLabel("〜"))
        row.addWidget(max_edit)

        self.filter_form.addRow("KPI Filter", row)

        self.filter_kpi_combo = combo
        self.filter_min_edit = min_edit
        self.filter_max_edit = max_edit

        combo.currentIndexChanged.connect(
            lambda _i: self._on_filter_kpi_changed(combo.currentData())
        )
        min_edit.textChanged.connect(lambda _t: self._on_filter_value_changed())
        max_edit.textChanged.connect(lambda _t: self._on_filter_value_changed())

        # 初期状態（フィルタなし）
        self._apply_filter_from_ui()

    # ---- テーブル再構築 ----
    def _rebuild_table(self, show_all: bool):
        if self.df_all is None or self.df_all.empty:
            self.model = DataFrameModel(pd.DataFrame())
            self.proxy = RangeFilterProxy([])
            self.proxy.setSourceModel(self.model)
            self.table.setModel(self.proxy)
            return

        if "__row_id" not in self.df_all.columns:
            self.df_all = self.df_all.copy()
            self.df_all["__row_id"] = range(len(self.df_all))

        df_current = self.df_all.copy()

        if show_all:
            all_cols = [
                c for c in df_current.columns
                if not (c.startswith("imputed__") or c == "__row_id")
            ]

            priority: list[str] = []
            priority += [c for c in NAME_COLUMNS if c in all_cols]
            priority += self._display_kpi_columns()

            extras = [c for c in all_cols if c not in priority]

            seen = set()
            cols: list[str] = []
            for c in priority + extras:
                if c not in seen:
                    cols.append(c)
                    seen.add(c)
        else:
            cols = NAME_COLUMNS + self._display_kpi_columns()
            cols = [
                c for c in cols
                if c in df_current.columns and not (c.startswith("imputed__") or c == "__row_id")
            ]
            if not cols:
                cols = [
                    c for c in df_current.columns
                    if not (c.startswith("imputed__") or c == "__row_id")
                ]

        df_view = df_current[cols].copy()

        # 赤字用マスク
        mask_view = None
        flag_map = {c: f"imputed__{c}" for c in cols}
        if any(fc in self.df_all.columns for fc in flag_map.values()):
            mask_view = pd.DataFrame(False, index=df_view.index, columns=df_view.columns)
            for c, fc in flag_map.items():
                if fc in self.df_all.columns:
                    mask_view[c] = (
                        self.df_all[fc].astype(bool).reindex(df_view.index).values
                    )

        self.model = DataFrameModel(df_view, mask_view)
        self.proxy = RangeFilterProxy(df_view.columns.tolist())
        self.proxy.setSourceModel(self.model)
        self.table.setModel(self.proxy)

        # 見た目調整
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Interactive)

        default_width = 160
        wider = {"first_name": 160, "last_name": 160}
        visible_cols = list(self.model._df.columns)
        for i, c in enumerate(visible_cols):
            self.table.setColumnWidth(i, wider.get(c, default_width))
        self.table.verticalHeader().setDefaultSectionSize(26)

        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # フィルタ再適用
        if hasattr(self, "_current_filter_col"):
            self.proxy.set_filter_column(self._current_filter_col)
            self._apply_filter_from_ui()

    # ---- フィルタ挙動 ----
    def _on_filter_kpi_changed(self, col_name: str | None):
        self._current_filter_col = col_name
        self._apply_filter_from_ui()

    def _on_filter_value_changed(self):
        self._apply_filter_from_ui()

    def _apply_filter_from_ui(self):
        if not hasattr(self, "proxy"):
            return

        col = self._current_filter_col
        if not col:
            self.proxy.set_filter_column(None)
            self.proxy.set_range(None, None)
            return

        if not self.filter_min_edit or not self.filter_max_edit:
            self.proxy.set_filter_column(None)
            self.proxy.set_range(None, None)
            return

        tmin = self.filter_min_edit.text().strip()
        tmax = self.filter_max_edit.text().strip()
        q1, q3 = self._filter_stats.get(col, (None, None))

        # ★ここがポイント：
        # 両方空欄で、Q1/Q3が計算できているときは
        # 実際にテキストボックスへQ1〜Q3を入れて見えるようにする
        if tmin == "" and tmax == "" and (q1 is not None or q3 is not None):
            self.filter_min_edit.blockSignals(True)
            self.filter_max_edit.blockSignals(True)
            if q1 is not None:
                self.filter_min_edit.setText(f"{q1:.3f}")
                tmin = self.filter_min_edit.text().strip()
            if q3 is not None:
                self.filter_max_edit.setText(f"{q3:.3f}")
                tmax = self.filter_max_edit.text().strip()
            self.filter_min_edit.blockSignals(False)
            self.filter_max_edit.blockSignals(False)

        def parse_or_default(text: str, default: float | None):
            if text == "":
                return default
            try:
                return float(text)
            except Exception:
                return default

        vmin = parse_or_default(tmin, q1)
        vmax = parse_or_default(tmax, q3)

        # Q1, Q3 どちらも取れない場合はフィルタ無し扱い
        if vmin is None and vmax is None:
            self.proxy.set_filter_column(None)
            self.proxy.set_range(None, None)
            return

        self.proxy.set_filter_column(col)
        self.proxy.set_range(vmin, vmax)


    # ---- モード変更 ----
    def _on_mode_changed(self, mode: str):
        if mode == getattr(self, "time_mode", None):
            return
        self.time_mode = mode
        self._settings.setdefault("ui", {})["time_mode"] = mode
        _save_json(SETTINGS_PATH, self._settings)

        self._build_filters()
        self._rebuild_table(self.debug_all_cols.isChecked())

    # ---- 戻る ----
    def go_back(self):
        sw = self.stacked_widget
        idx = sw.indexOf(self)
        if idx != -1:
            sw.removeWidget(self)
        sw.setCurrentIndex(0)

    # ---- 行削除 ----
    def delete_selected_rows(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            QMessageBox.information(self, "Delete", "Please select rows to delete")
            return

        if QMessageBox.question(
            self,
            "Delete Rows",
            f"{len(sel)} rows will be deleted. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) != QMessageBox.Yes:
            return

        source_rows = sorted({self.proxy.mapToSource(i).row() for i in sel})

        if "__row_id" in self.df_all.columns:
            row_ids = self.df_all.iloc[source_rows]["__row_id"].tolist()
            self.df_all.drop(
                self.df_all.index[self.df_all["__row_id"].isin(row_ids)], inplace=True
            )
        else:
            self.df_all.drop(self.df_all.index[source_rows], inplace=True)

        self.df_all.reset_index(drop=True, inplace=True)
        self._rebuild_table(self.debug_all_cols.isChecked())

    # ---- CSVエクスポート ----
    def export_current_view_to_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "kpi_export.csv", "CSV Files (*.csv)"
        )
        if not path:
            return

        rows = [
            self.proxy.mapToSource(self.proxy.index(r, 0)).row()
            for r in range(self.proxy.rowCount())
        ]
        export_df = self.df_all.iloc[rows].copy()

        drop_cols = [
            c for c in export_df.columns if c.startswith("imputed__")
        ] + ["__row_id"]
        export_df.drop(
            columns=[c for c in drop_cols if c in export_df.columns],
            inplace=True,
            errors="ignore",
        )

        try:
            export_df.to_csv(path, index=False, encoding="utf-8-sig")
            QMessageBox.information(self, "Export", f"CSV Saved:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", f"Save failed:\n{e}")

    # ---- 右クリックメニュー / クリップボードコピー ----
    def _on_table_context_menu(self, pos: QPoint):
        tv = self.table
        idx = tv.indexAt(pos)
        sel = tv.selectionModel()

        if idx.isValid() and not sel.isSelected(idx):
            tv.selectRow(idx.row())

        menu = QMenu(tv)
        act_copy = QAction("選択行をコピー（TSV）", menu)
        act_copy.triggered.connect(self._copy_selected_rows_to_clipboard)
        menu.addAction(act_copy)
        menu.exec_(tv.viewport().mapToGlobal(pos))

    def _copy_selected_rows_to_clipboard(self):
        tv = self.table
        model = tv.model()
        sel = tv.selectionModel()
        if not sel or not sel.hasSelection():
            return

        visible_cols = []
        headers = []
        for c in range(model.columnCount()):
            if not tv.isColumnHidden(c):
                visible_cols.append(c)
                hdr = model.headerData(c, Qt.Horizontal)
                headers.append("" if hdr is None else str(hdr))

        rows = sorted({i.row() for i in sel.selectedIndexes()})
        if not rows:
            rows = sorted(i.row() for i in sel.selectedRows())

        lines = ["\t".join(headers)]
        for r in rows:
            vals = []
            for c in visible_cols:
                idx = model.index(r, c)
                text = model.data(idx, Qt.DisplayRole)
                vals.append("" if text is None else str(text))
            lines.append("\t".join(vals))

        tsv = "\n".join(lines)
        QGuiApplication.clipboard().setText(tsv)
        print(f"[KPI] {len(rows)} 行をクリップボードへコピーしました")

    # ---- リロード ----
    def _reload_kpi(self):
        print("[KPI] リロード開始")

        # 現在のソートを覚えておく
        hh = self.table.horizontalHeader()
        sort_col = hh.sortIndicatorSection()
        sort_ord = hh.sortIndicatorOrder()

        # データを再読込
        self._load_data_and_prepare_kpi()

        # フィルタはモードに合わせて作り直し
        self._build_filters()
        self._rebuild_table(self.debug_all_cols.isChecked())

        # ソート復元
        try:
            self.table.sortByColumn(sort_col, sort_ord)
        except Exception:
            pass

        print("[KPI] リロード完了")

    def _reload_kpi_json(self):
        print("[KPI] kpi.json リロード開始")

        # kpi.json を読み直し
        self._interval_config = _load_kpi_intervals(KPI_INTERVALS_PATH)

        # 新しい定義に基づいて区間列を追加（既にある列はスキップされる）
        self._ensure_interval_columns()

        # フィルタとテーブルを作り直し
        self._build_filters()
        self._rebuild_table(self.debug_all_cols.isChecked())

        print("[KPI] kpi.json リロード完了")

    def _open_kpi_json_editor(self):
        """kpi.json編集ページへ遷移"""
        editor = KPIJsonEditorPage(self)
        self.stacked_widget.addWidget(editor)
        self.stacked_widget.setCurrentWidget(editor)
