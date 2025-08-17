# kpi_page.py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QHBoxLayout, QTableView, QFormLayout, QCheckBox
)
from PyQt5.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
)
from PyQt5.QtGui import QDoubleValidator, QBrush, QColor
import math
from utils import fetch_df_from_db
import datetime
import numpy as np
import pandas as pd

# 表示列
DISPLAY_COLUMNS = [
    "first_name", "last_name",
    "entry_speed", "jump_speed",
    "Time000to100", "Time100to200", "Time000to200"
]
NUMERIC_COLUMNS = [
    "entry_speed", "jump_speed",
    "Time000to100", "Time100to200", "Time000to200"
]
NAME_COLUMNS = ["first_name", "last_name"]

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

            # --- pandas.Timestamp / datetime.datetime / numpy.datetime64 → 時刻だけ表示 ---
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
                ms = ts.microsecond // 1000
                return ts.strftime("%H:%M:%S") if ms == 0 else ts.strftime("%H:%M:%S.%f")[:-3]

            # --- pandas.Timedelta → mm:ss.mmm 表示（負号対応）---
            if isinstance(val, pd.Timedelta):
                total_ms = int(val / pd.Timedelta(milliseconds=1))
                sign = "-" if total_ms < 0 else ""
                total_ms = abs(total_ms)
                minutes, rem = divmod(total_ms, 60_000)
                seconds, ms = divmod(rem, 1000)
                return f"{sign}{minutes:02d}:{seconds:02d}.{ms:03d}"

            # それ以外は通常表示
            return str(val)

        # ② 補完セルなら赤字
        if role == Qt.ForegroundRole and self._mask is not None:
            try:
                if bool(self._mask.iat[index.row(), index.column()]):
                    return QBrush(QColor(220, 0, 0))
            except Exception:
                pass    

        # ③ 編集用値（既存通り）
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
    def __init__(self, column_names):
        super().__init__()
        self.column_names = column_names  # 表示列名の並び
        self.ranges = {name: (None, None) for name in NUMERIC_COLUMNS}  # name -> (min,max)

    def setRange(self, col_name: str, vmin: str, vmax: str):
        def to_float_or_none(s):
            s = s.strip()
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
        # 数値列に対して min/max のAND条件
        for col_name, (vmin, vmax) in self.ranges.items():
            if col_name not in self.column_names:
                continue
            col_idx = self.column_names.index(col_name)
            idx = model.index(source_row, col_idx)
            text = model.data(idx, Qt.DisplayRole)
            # 空や非数値は、レンジ指定がある時は落とす・無い時は通す
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

        self.setWindowTitle("KPI 表示")
        self.resize(980, 640)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("KPI結果（数値レンジフィルタ対応）"))

        # ▼ 追加: 全列表示トグル（初期ON=デバッグ表示）
        self.debug_all_cols = QCheckBox("全列表示（デバッグ）")
        self.debug_all_cols.setChecked(True)  # まずは制限なしで確認したいという要望に沿う
        layout.addWidget(self.debug_all_cols)

        # --- DB取得（クエリは変更しない）---
        df_any, _users = fetch_df_from_db(query)

        # ▼ 元データは保持（切替で使い回す）
        self.df_all = df_any.copy()

        # ▼ Time100→200 の表記ゆれ（utils側が "Time1000to200" を生成）に当面対応
        #   表示やフィルタ指定では "Time100to200" をラベルとして使い、
        #   実体が存在すれば "Time1000to200" を参照する
        self.alias_map = {"Time100to200": "Time100to200"}

        # 必要列の補完（無ければ NaN/空文字を立てておく）
        for c in NAME_COLUMNS:
            if c not in self.df_all.columns:
                self.df_all[c] = ""
        for c in NUMERIC_COLUMNS:
            # 実体列名を解決して存在チェック
            real_c = self.alias_map.get(c, c)
            if real_c not in self.df_all.columns:
                self.df_all[real_c] = float("nan")

        # --- フィルタUI（最小・最大） ---
        form = QFormLayout()
        self.min_edits = {}
        self.max_edits = {}
        validator = QDoubleValidator()

        def add_min_max_row(label_text, real_col_name):
            row = QHBoxLayout()
            min_edit = QLineEdit(); min_edit.setPlaceholderText("min"); min_edit.setValidator(validator)
            max_edit = QLineEdit(); max_edit.setPlaceholderText("max"); max_edit.setValidator(validator)
            # self.proxy は後で作り直すので、呼ばれた時点の self.proxy を参照する
            min_edit.textChanged.connect(lambda _t: self.proxy.setRange(real_col_name, min_edit.text(), max_edit.text()))
            max_edit.textChanged.connect(lambda _t: self.proxy.setRange(real_col_name, min_edit.text(), max_edit.text()))
            row.addWidget(min_edit); row.addWidget(QLabel("〜")); row.addWidget(max_edit)
            form.addRow(label_text, row)
            self.min_edits[real_col_name] = min_edit
            self.max_edits[real_col_name] = max_edit

        # 実体の列名（存在すれば alias を解決）
        def resolve(col_name, columns):
            cand = self.alias_map.get(col_name, col_name)
            return cand if cand in columns else col_name  # 無ければ元名のまま（後段でガード済）

        add_min_max_row("Entry Speed",  resolve("entry_speed",  self.df_all.columns))
        add_min_max_row("Jump Speed",   resolve("jump_speed",   self.df_all.columns))
        add_min_max_row("Time 000→100", resolve("Time000to100", self.df_all.columns))
        add_min_max_row("Time 100→200", resolve("Time100to200", self.df_all.columns))  # 実体は Time1000to200 かも
        add_min_max_row("Time 000→200", resolve("Time000to200", self.df_all.columns))

        layout.addLayout(form)

        # ▼ min/max 初期値の自動セット（存在する数値列のみ）
        present_num_cols = []
        for c in NUMERIC_COLUMNS:
            rc = resolve(c, self.df_all.columns)
            if rc in self.df_all.columns and rc not in present_num_cols:
                present_num_cols.append(rc)

        if present_num_cols:
            num_df = self.df_all[present_num_cols].apply(pd.to_numeric, errors="coerce")
            col_mins = num_df.min(skipna=True)
            col_maxs = num_df.max(skipna=True)
            for rc in present_num_cols:
                vmin = col_mins.get(rc); vmax = col_maxs.get(rc)
                if pd.notna(vmin):
                    self.min_edits[rc].blockSignals(True); self.min_edits[rc].setText(f"{float(vmin):.3f}"); self.min_edits[rc].blockSignals(False)
                if pd.notna(vmax):
                    self.max_edits[rc].blockSignals(True); self.max_edits[rc].setText(f"{float(vmax):.3f}"); self.max_edits[rc].blockSignals(False)

        # --- テーブル（ソート可） ---
        self.table = QTableView()
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        # ▼ 表の初期構築（デバッグ=全列表示）
        self._rebuild_table(show_all=True)

        # ▼ トグル切替で再構築
        self.debug_all_cols.stateChanged.connect(lambda _s: self._rebuild_table(self.debug_all_cols.isChecked()))

        # 戻る
        back_btn = QPushButton("← 戻る（メインへ）")
        back_btn.clicked.connect(self.go_back)
        layout.addWidget(back_btn)

        self.setLayout(layout)

    # ▼ 表の作り直し（全列 or 標準列）
    def _rebuild_table(self, show_all: bool):
        # 1) 表示列（imputed__* は表示しない）
        if show_all:
            cols = [c for c in self.df_all.columns if not c.startswith("imputed__")]
        else:
            cols = [c for c in DISPLAY_COLUMNS if c in self.df_all.columns and not c.startswith("imputed__")]
            if not cols:
                cols = [c for c in self.df_all.columns if not c.startswith("imputed__")]

        df_view = self.df_all[cols].copy()

        # 2) マスクを組み立て
        mask_view = pd.DataFrame(False, index=df_view.index, columns=df_view.columns)
        for c in cols:
            fcol = f"imputed__{c}"
            if fcol in self.df_all.columns:
                mask_view[c] = self.df_all[fcol].astype(bool).reindex(df_view.index).values

        # 3) モデルへ
        self.model = DataFrameModel(df_view, mask_view)
        self.proxy = RangeFilterProxy(cols)
        self.proxy.setSourceModel(self.model)
        self.table.setModel(self.proxy)

 
    def go_back(self):
        """メインに戻る。動的に積み上がるのを避けるため、このページをスタックから外す。"""
        sw = self.stacked_widget
        idx = sw.indexOf(self)
        if idx != -1:
            sw.removeWidget(self)
        sw.setCurrentIndex(0)