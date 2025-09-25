# kpi_page.py (fixed full)
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QHBoxLayout, QTableView, QFormLayout, QCheckBox,
    QAbstractItemView, QMessageBox, QFileDialog, QRadioButton, QButtonGroup,QHeaderView
)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
from PyQt5.QtGui import QDoubleValidator, QBrush, QColor
import math
import datetime
import numpy as np
import pandas as pd
import json
import os
from utils import fetch_df_from_db

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



# ---- 固定の名前列 ----
NAME_COLUMNS = ["first_name", "last_name", "Date", "FP_start"]

# ---- モード別・標準表示KPI列（存在しない列は自動スキップ）----
# RS: 000→625, 625→125, 000→125
# FLY: 000→100, 100→200, 000→200
KPI_BY_MODE = {
    "RS":  ["Time000to625", "Time625to125", "Time000to125"],
    "FLY": ["Time000to100", "Time100to200", "Time000to200"],
}

# フィルタUIに表示するラベル（見出し）
FILTER_LABELS = {
    "Time000to625": "Time 000→625",
    "Time625to125": "Time 625→125",
    "Time000to125": "Time 000→125",
    "Time000to100": "Time 000→100",
    "Time100to200": "Time 100→200",
    "Time000to200": "Time 000→200",
}

# ---- DataFrame -> Qt Model ----
class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, mask: pd.DataFrame = None):
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

            # 時刻だけ表示
            if isinstance(val, pd.Timestamp):
                ms = val.microsecond // 1000
                return val.strftime("%H:%M:%S") if ms == 0 else val.strftime("%H:%M:%S.%f")[:-3]

            if isinstance(val, datetime.datetime):
                ms = val.microsecond // 1000
                return val.strftime("%H:%M:%S") if ms == 0 else val.strftime("%H:%M:%S.%f")[:-3]

            if isinstance(val, np.datetime64):
                ts = pd.to_datetime(val)
                if pd.isna(ts):
                    return ""
                # pandasのTimestampに変換して同様に表示
                ts = pd.Timestamp(ts)
                ms = ts.microsecond // 1000
                return ts.strftime("%H:%M:%S") if ms == 0 else ts.strftime("%H:%M:%S.%f")[:-3]

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

# ---- 数値レンジ（min/max）で列ごとフィルタ ----
class RangeFilterProxy(QSortFilterProxyModel):
    """
    可視テーブル（source model）の列順 = visible_columns。
    フィルタ対象列 = self.ranges のキー（列名）。
    """
    def __init__(self, visible_columns):
        super().__init__()
        self.visible_columns = list(visible_columns)  # ソースモデルの列順
        self.ranges = {}  # name -> (min,max)

    def set_visible_columns(self, visible_columns):
        self.visible_columns = list(visible_columns)
        self.invalidateFilter()

    def setRange(self, col_name: str, vmin: str, vmax: str):
        def to_float_or_none(s):
            if s is None:
                return None
            s = str(s).strip()
            if s == "":
                return None
            try:
                return float(s)
            except Exception:
                return None
        self.ranges[col_name] = (to_float_or_none(vmin), to_float_or_none(vmax))
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        # 数値列のAND条件
        for col_name, (vmin, vmax) in self.ranges.items():
            if col_name not in self.visible_columns:
                # 今の表示テーブルにこの列が無ければ無視
                continue
            try:
                col_idx = self.visible_columns.index(col_name)
            except ValueError:
                continue
            idx = model.index(source_row, col_idx)
            text = model.data(idx, Qt.DisplayRole)
            try:
                val = float(text) if text not in (None, "") else math.nan
            except Exception:
                val = math.nan

            if vmin is not None:
                if math.isnan(val) or val < vmin:
                    return False
            if vmax is not None:
                if math.isnan(val) or val > vmax:
                    return False
        return True

class KPIPage(QWidget):
    def __init__(self, query, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self._settings = _load_json(SETTINGS_PATH)

        self.setWindowTitle("Display KPIs")
        self.resize(980, 640)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("KPIs"))

        # --- トグル & モード切替 ---
        row_top = QHBoxLayout()
        self.debug_all_cols = QCheckBox("Show All Columns")
        self.debug_all_cols.setChecked(False)
        row_top.addWidget(self.debug_all_cols)

        self.mode_group = QButtonGroup(self)
        self.rb_rs = QRadioButton("Rolling/Standing")
        self.rb_fly = QRadioButton("Flying")
        self.mode_group.addButton(self.rb_rs)
        self.mode_group.addButton(self.rb_fly)
        self.rb_rs.setChecked(True)  # 既定はRS
        row_top.addWidget(self.rb_rs)
        row_top.addWidget(self.rb_fly)
        layout.addLayout(row_top)

        # --- DB取得 ---
        df_any, _users = fetch_df_from_db(query)
        df_any = df_any.copy()
        if "__row_id" not in df_any.columns:
            df_any["__row_id"] = range(len(df_any))
        self.df_all = df_any

        # 数値フィルタフォーム（後で差し替え可能にする）
        self.filter_form = QFormLayout()
        layout.addLayout(self.filter_form)
        self.min_edits = {}
        self.max_edits = {}

        # --- テーブル ---
        self.table = QTableView()
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        # ボタン類
        btn_row_delete = QPushButton("Delete selected rows")
        btn_export_csv = QPushButton("Export to CSV")
        btn_row_delete.clicked.connect(self.delete_selected_rows)
        btn_export_csv.clicked.connect(self.export_current_view_to_csv)
        btn_box = QHBoxLayout()
        btn_box.addWidget(btn_row_delete)
        btn_box.addWidget(btn_export_csv)
        layout.addLayout(btn_box)

        # 戻る
        back_btn = QPushButton("← Back to Main")
        back_btn.clicked.connect(self.go_back)
        layout.addWidget(back_btn)

        self.setLayout(layout)

        # 初期状態
        self.time_mode = "RS"   # or "FLY"
        self._build_filters()   # モードに応じたフィルタ欄
        self._rebuild_table(self.debug_all_cols.isChecked())

        # トグル/モードのイベント
        self.debug_all_cols.stateChanged.connect(lambda _s: self._rebuild_table(self.debug_all_cols.isChecked()))
        self.rb_rs.toggled.connect(lambda checked: checked and self._on_mode_changed("RS"))
        self.rb_fly.toggled.connect(lambda checked: checked and self._on_mode_changed("FLY"))

    # ---- ヘルパ：現在のモードに対応する「標準表示KPI列」を返す（存在するものだけ）----
    def _current_kpi_columns(self):
        """
        KPIフィルタ対象の列を、モード(self.time_mode)に応じて最小集合に絞る。
        R/S:   000-125 だけ
        FLY:   000-200 だけ
        ※復活用の候補はコメントで残す
        """
        mode = (getattr(self, "time_mode", "FLY") or "FLY").upper()

        if mode == "RS":
            allowed = ["Time000to125"]
            # 復活候補:
            # allowed = ["Time000to625", "Time625to125", "Time000to125"]
        else:  # FLY (既定)
            allowed = ["Time000to200"]
            # 復活候補:
            # allowed = ["Time000to100", "Time100to200", "Time000to200"]

        # 実際にデータフレームに存在する列だけ返す
        cols_in_df = [c for c in allowed if c in getattr(self, "df_all", {}).columns]
        return cols_in_df

    def _display_kpi_columns(self):
        # 表示用（3列）: モード別の標準KPI
        mode = (getattr(self, "time_mode", "RS") or "RS").upper()
        if mode == "RS":
            allowed = ["Time000to625", "Time625to125", "Time000to125"]
        else:  # FLY
            allowed = ["Time000to100", "Time100to200", "Time000to200"]
        return [c for c in allowed if c in self.df_all.columns]

    def _filter_kpi_columns(self):
        # フィルタUI用（1列）
        mode = (getattr(self, "time_mode", "RS") or "RS").upper()
        if mode == "RS":
            return ["Time000to125"]
        else:  # FLY
            return ["Time000to200"]

    # ---- フィルタUIをビルド（モードごとに作り直し）----
    def _build_filters(self):
        # 既存行をクリア
        while self.filter_form.rowCount():
            self.filter_form.removeRow(0)
        self.min_edits.clear()
        self.max_edits.clear()

        # 数値列（モードの標準KPI列のみ）
        numeric_cols = self._filter_kpi_columns()

        validator = QDoubleValidator()
        for col in numeric_cols:
            label_text = FILTER_LABELS.get(col, col)
            row = QHBoxLayout()
            min_edit = QLineEdit(); min_edit.setPlaceholderText("min"); min_edit.setValidator(validator)
            max_edit = QLineEdit(); max_edit.setPlaceholderText("max"); max_edit.setValidator(validator)
            # 変更時：プロキシへ適用＆設定ファイルにも保存
            min_edit.textChanged.connect(lambda _t, c=col, me=min_edit, xe=max_edit: self._on_filter_changed(c, me.text(), xe.text()))
            max_edit.textChanged.connect(lambda _t, c=col, me=min_edit, xe=max_edit: self._on_filter_changed(c, me.text(), xe.text()))
            row.addWidget(min_edit); row.addWidget(QLabel("〜")); row.addWidget(max_edit)
            self.filter_form.addRow(label_text, row)
            self.min_edits[col] = min_edit
            self.max_edits[col] = max_edit

        # 初期値の適用：setting.json に保存済みがあれば優先し、無ければ自動計算した値を使う
        if numeric_cols:
            # 自動計算の下限/上限
            num_df = self.df_all[numeric_cols].apply(pd.to_numeric, errors="coerce")
            auto_min = num_df.min(skipna=True)
            auto_max = num_df.max(skipna=True)
            # 設定から取得
            fr = self._settings.get("filter_ranges", {}).get(self.time_mode, {})
            for c in numeric_cols:
                saved = fr.get(c, {})
                vmin = saved.get("min", None)
                vmax = saved.get("max", None)
                # 文字列をそのまま使う（空文字も可）。無ければ自動推定値を入れて見やすく。
                if vmin is None and pd.notna(auto_min.get(c)):
                    vmin = f"{float(auto_min.get(c)):.3f}"
                if vmax is None and pd.notna(auto_max.get(c)):
                    vmax = f"{float(auto_max.get(c)):.3f}"
                # UIへ反映（シグナルは一旦止める）
                self.min_edits[c].blockSignals(True); self.min_edits[c].setText("" if vmin is None else str(vmin)); self.min_edits[c].blockSignals(False)
                self.max_edits[c].blockSignals(True); self.max_edits[c].setText("" if vmax is None else str(vmax)); self.max_edits[c].blockSignals(False)
                # プロキシへも反映
                if hasattr(self, "proxy"):
                    self.proxy.setRange(c, self.min_edits[c].text(), self.max_edits[c].text())

    # ---- 表の作り直し（全列 or 標準列）----
    def _rebuild_table(self, show_all: bool):
        if self.df_all is None:
            self.model = DataFrameModel(pd.DataFrame())
            self.proxy = RangeFilterProxy([])
            self.proxy.setSourceModel(self.model)
            self.table.setModel(self.proxy)
            return

        if "__row_id" not in self.df_all.columns:
            self.df_all = self.df_all.copy()
            self.df_all["__row_id"] = range(len(self.df_all))

        if show_all:
            cols = [c for c in self.df_all.columns if not (c.startswith("imputed__") or c == "__row_id")]
        else:
            # 名称列 + モード別KPI列（存在するものだけ）
            cols = NAME_COLUMNS + self._display_kpi_columns()
            cols = [c for c in cols if c in self.df_all.columns and not (c.startswith("imputed__") or c == "__row_id")]
            if not cols:
                # 何も無ければ全列にフォールバック
                cols = [c for c in self.df_all.columns if not (c.startswith("imputed__") or c == "__row_id")]

        df_view = self.df_all[cols].copy()

        # 赤字用マスク
        mask_view = None
        flag_map = {c: f"imputed__{c}" for c in cols}
        if any(fc in self.df_all.columns for fc in flag_map.values()):
            mask_view = pd.DataFrame(False, index=df_view.index, columns=df_view.columns)
            for c, fc in flag_map.items():
                if fc in self.df_all.columns:
                    mask_view[c] = self.df_all[fc].astype(bool).reindex(df_view.index).values

        # モデル/プロキシ
        self.model = DataFrameModel(df_view, mask_view)
        self.proxy = RangeFilterProxy(df_view.columns.tolist())
        self.proxy.setSourceModel(self.model)
        self.table.setModel(self.proxy)
        
        # 列ヘッダを手動調整可能に
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Interactive)

        # 全列の基準幅（ここを好みで 140〜160 に）
        default_width = 160
        wider = {
            "first_name": 160,
            "last_name": 160,
            # "Time000to200": 150,
            # "Time000to125": 150,
        }
        
        visible_cols = list(self.model._df.columns)  # DataFrameModel 内部DFの列
        for i, c in enumerate(visible_cols):
            self.table.setColumnWidth(i, wider.get(c, default_width))

        # 行の高さも少し余裕を（任意）
        self.table.verticalHeader().setDefaultSectionSize(26)

        # 現在のフィルタ入力をプロキシへ反映（モードで列が変わった時のため）
        # ※ visible_columns は df_view.columns。フィルタは self.min_edits の列だけ設定
        for c, me in self.min_edits.items():
            xe = self.max_edits[c]
            self.proxy.setRange(c, me.text(), xe.text())

        # 複数行選択
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)

    # ---- モード変更 ----
    def _on_mode_changed(self, mode: str):
        if mode == getattr(self, "time_mode", None):
            return
        self.time_mode = mode
        self._settings.setdefault("ui", {})["time_mode"] = mode
        _save_json(SETTINGS_PATH, self._settings)
        # フィルタ欄を作り直し（モードの標準KPI列に合わせる）
        self._build_filters()
        # テーブルを再構築
        self._rebuild_table(self.debug_all_cols.isChecked())

    def go_back(self):
        sw = self.stacked_widget
        idx = sw.indexOf(self)
        if idx != -1:
            sw.removeWidget(self)
        sw.setCurrentIndex(0)

    def delete_selected_rows(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            QMessageBox.information(self, "Delete", "Please select rows to delete")
            return
        if QMessageBox.question(
            self, "Delete Rows",
            f"{len(sel)} rows will be deleted. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        ) != QMessageBox.Yes:
            return

        # プロキシ行 → ソース行
        source_rows = sorted({ self.proxy.mapToSource(i).row() for i in sel })

        # __row_id が無い環境でも安全に落ちないように保険
        if "__row_id" in self.df_all.columns:
            # __row_id で消す
            row_ids = self.df_all.iloc[source_rows]["__row_id"].tolist()
            self.df_all.drop(self.df_all.index[self.df_all["__row_id"].isin(row_ids)], inplace=True)
        else:
            # 素直に index で消す（並びに注意）
            self.df_all.drop(self.df_all.index[source_rows], inplace=True)

        self.df_all.reset_index(drop=True, inplace=True)
        self._rebuild_table(self.debug_all_cols.isChecked())

    def export_current_view_to_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "kpi_export.csv", "CSV Files (*.csv)")
        if not path:
            return

        # フィルタ後の行だけ抽出し、列は df_all の“全列”で出力
        rows = [ self.proxy.mapToSource(self.proxy.index(r, 0)).row()
                 for r in range(self.proxy.rowCount()) ]
        export_df = self.df_all.iloc[rows].copy()

        # 内部列があれば除外（保険）
        drop_cols = [c for c in export_df.columns if c.startswith("imputed__")] + ["__row_id"]
        export_df.drop(columns=[c for c in drop_cols if c in export_df.columns],
                       inplace=True, errors="ignore")

        try:
            export_df.to_csv(path, index=False, encoding="utf-8-sig")
            QMessageBox.information(self, "Export", f"CSV Saved:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", f"Save failed:\n{e}")

    def _on_filter_changed(self, col_name: str, vmin: str, vmax: str):
        # プロキシへ反映
        if hasattr(self, "proxy"):
            self.proxy.setRange(col_name, vmin, vmax)
        # 設定に保存
        if "filter_ranges" not in self._settings:
            self._settings["filter_ranges"] = {}
        if self.time_mode not in self._settings["filter_ranges"]:
            self._settings["filter_ranges"][self.time_mode] = {}
        self._settings["filter_ranges"][self.time_mode][col_name] = {
            "min": vmin if vmin is not None else "",
            "max": vmax if vmax is not None else "",
        }
        _save_json(SETTINGS_PATH, self._settings)
