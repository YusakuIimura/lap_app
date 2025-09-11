import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# ここはセキュリティ的に書き換えるべき
PASS="cyvri5-Funcyb-cesfuk"
USER="readonly"

translate_dict ={
    '00042056': '100m',
    '00042616': '150m',
    '000427a3': '50m',
    '000427bc': '0m',
    '000427be': 'AP2',
    '0004282a': 'AP1',
    '0004283c': 'FP',
    '00042899': '60m',
    '000428a2': '200m',
    '000428a3': 'BP',
    '10044ec3': 'SB1',  # Starting Block1
    '20044001': 'SW1',  # Stop Watch 1
    '20044002': 'SW2',  # Stop Watch 2
    '200475ca': 'SW3',  # FIXME
    '00042b8f': 'PL1',  # Portable timing system Loop 1
    '00042b55': 'PL2',  # Portable timing system Loop 2
    '00042b4e': 'PL3',  # Portable timing system Loop 3
    '00042b2e': 'PL4',  # Portable timing system Loop 4
}

POINT_ORDER = ["FP","0m","60m","AP1","50m","100m","BP","150m","AP2","200m","FP_END"]

# 隣接点の距離[m]
SEGMENTS = [
    ("FP_start", "0m_start", 17.97),
    ("0m_start", "60m",      42.03),
    ("60m",      "AP1",       2.50),
    ("AP1",      "50m",       5.47),
    ("50m",      "100m",     50.00),
    ("100m",     "BP",        7.03),
    ("BP",       "150m",     42.97),
    ("150m",     "AP2",      19.53),
    ("AP2",      "200m",     30.47),
    ("200m",     "FP_END",   32.03),
]

def _build_distance_map():
    """SEGMENTSから列順と累積距離[m]を算出"""
    order = ["FP_start"]
    dist = {"FP_start": 0.0}
    for a, b, d in SEGMENTS:
        # a の累積距離が未登録なら直前の値を使う（通常は登録済み）
        if a not in dist:
            dist[a] = dist[order[-1]]
        dist[b] = dist[a] + d
        if b not in order:
            order.append(b)
    return order, dist

def impute_times_by_distance(df_laps: pd.DataFrame) -> pd.DataFrame:
    """
    ラップ内タイムの欠損を距離比例で補完（外挿なし）。
    併せて各タイム列ごとに imputed__{col} のブール列を付与して、どこを埋めたか保持する。
    """
    cols_order, dist_map = _build_distance_map()  # ["FP_start","0m_start",...,"200m","FP_END"]
    df = df_laps.copy()

    # 終端（次ラップのFP）を補完用に利用
    df["FP_END"] = df.get("FP_2nd", pd.NaT)

    # 各列のフラグ列を初期化（False）
    flag_cols = [c for c in cols_order if c != "FP_END"]
    for c in flag_cols:
        df[f"imputed__{c}"] = False

    for i in df.index:
        times = df.loc[i, cols_order].copy()

        # 既知インデックス
        known_idx = [k for k, c in enumerate(cols_order) if pd.notna(times[c])]
        if len(known_idx) < 2:
            continue

        # 既知のペアごとに距離比例で線形補間（挟まれている欠損のみ）
        for a, b in zip(known_idx[:-1], known_idx[1:]):
            colL, colR = cols_order[a], cols_order[b]
            tL, tR = times[colL], times[colR]
            if not (pd.notna(tL) and pd.notna(tR) and tR > tL):
                continue

            dL, dR = dist_map[colL], dist_map[colR]
            span = dR - dL
            if span <= 0:
                continue

            for k in range(a + 1, b):
                ck = cols_order[k]
                if ck == "FP_END":
                    continue
                if pd.isna(times[ck]):
                    frac = (dist_map[ck] - dL) / span
                    times[ck] = tL + (tR - tL) * frac
                    df.at[i, f"imputed__{ck}"] = True  # ← 補完フラグON

        # 反映
        df.loc[i, cols_order] = times

    # 作業列は落とす
    df = df.drop(columns=["FP_END"])
    return df

CUM_DIST = {"FP": 0.0}
_total = 0.0
for a, b, d in SEGMENTS:
    _total += d
    CUM_DIST[b] = _total
LAP_LENGTH = CUM_DIST["FP_END"]

# EC2経由でDBへアクセス
# def get_df_from_db(query, database="trackview"):
#     engine = create_engine(
#         f"mysql+pymysql://{USER}:{PASS}@127.0.0.1:3307/{database}"
#     )
#     with engine.connect() as conn:
#         df = pd.read_sql(query, conn)
#     return df

def get_df_from_db(query, database="trackview"):
    host = "trackview-instance-1.chakc5iaoqqz.ap-northeast-1.rds.amazonaws.com"
    port = 3306
    engine = create_engine(
        f"mysql+pymysql://{USER}:{PASS}@{host}:{port}/{database}"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def split_laps(df):
    """
    Robust lap splitter:
      - ラップ開始: その周で最初に現れた 'FP' または '0m' をアンカーとして採用
      - ラップ確定: 同じアンカーが再び来たタイミング（FP→…→FP / 0m→…→0m）
      - 各計測点はそのラップで最初の観測だけ記録
      - FPは2段構え：
          FP_first : ラップ内で最初のFP（生値, 内部）
          FP_start : 表示/計算用。0mより後に現れたFPはNaTにして負のentry_speedを回避
      - _2nd列は確定後にシフトで一括:
          FP_2nd = 次行の FP_first
          0m_2nd = 次行の 0m_start
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    anchors = {"FP", "0m"}
    inner_pts = ["60m", "AP1", "50m", "100m", "BP", "150m", "AP2", "200m"]

    # 中間生成用の列（FP_firstは内部用）
    tmp_cols = ["FP_first", "0m_start"] + inner_pts
    rows = []

    cur = {k: None for k in tmp_cols}
    anchor = None  # 現在のラップのアンカー: 'FP' or '0m' or None

    def flush():
        nonlocal cur, rows, anchor
        if any(v is not None for v in cur.values()):
            rows.append(cur)
        cur = {k: None for k in tmp_cols}
        # 次のラップ開始時に再設定する
        anchor = None

    for _, r in df.iterrows():
        pos = r["position"]
        ts  = r["timestamp"]

        if pos not in anchors and pos not in inner_pts:
            continue

        if pos in anchors:
            # アンカー処理
            if anchor is None:
                # 新規ラップ開始
                anchor = pos
                if pos == "FP":
                    if cur["FP_first"] is None:
                        cur["FP_first"] = ts
                else:  # pos == "0m"
                    if cur["0m_start"] is None:
                        cur["0m_start"] = ts
            else:
                if pos == anchor:
                    # 同じアンカーが再来 → 現ラップ確定＆新ラップ開始
                    flush()
                    anchor = pos
                    if pos == "FP":
                        cur["FP_first"] = ts
                    else:  # pos == "0m"
                        cur["0m_start"] = ts
                else:
                    # 異なるアンカーがラップ途中で現れた → 値だけ記録して継続
                    if pos == "FP" and cur["FP_first"] is None:
                        cur["FP_first"] = ts
                    if pos == "0m" and cur["0m_start"] is None:
                        cur["0m_start"] = ts
            continue

        # 内側ポイント：最初の観測だけ記録
        if pos in inner_pts and cur[pos] is None:
            cur[pos] = ts

    # 末尾も確定
    flush()

    if not rows:
        return pd.DataFrame(columns=[
            "FP_start", "0m_start", *inner_pts, "FP_2nd", "0m_2nd"
        ])

    out_tmp = pd.DataFrame(rows, columns=tmp_cols)

    # 表示/計算用の FP_start を作る（0mより後のFPは欠測扱いにして負のentry_speedを防止）
    def compute_fp_start(row):
        fp = row["FP_first"]
        z0 = row["0m_start"]
        if pd.isna(fp):
            return pd.NaT
        if pd.isna(z0):
            return fp
        return fp if fp <= z0 else pd.NaT  # 0mより後に来たFPは使わない

    out = pd.DataFrame()
    out["FP_start"] = out_tmp.apply(compute_fp_start, axis=1)
    out["0m_start"] = out_tmp["0m_start"]
    for c in inner_pts:
        out[c] = out_tmp[c]

    # _2nd は「次行の start」をシフトで付与
    out["FP_2nd"] = out_tmp["FP_first"].shift(-1)   # 表示用で欠測にしたFPでも、次FPの実観測時刻は保持
    out["0m_2nd"] = out["0m_start"].shift(-1)
    
    # 年月日列（ラップ内で最初に観測できた時刻の日付）
    cols_for_date = ["FP_first", "0m_start"] + inner_pts  # inner_pts = ["60m","AP1","50m","100m","BP","150m","AP2","200m"]
    first_ts = out_tmp[cols_for_date].apply(
        lambda r: r.dropna().min() if r.notna().any() else pd.NaT, axis=1
    )
    out.insert(0, "Date", pd.to_datetime(first_ts).dt.date)  # 先頭列に挿入

    return out

def calculate_entry_speed(row):
    l = 17.97  # 0m-FP間距離（m）
    t1, t2 = row.get("0m_start"), row.get("FP_start")
    if pd.isna(t1) or pd.isna(t2):
        return np.nan
    td = t1 - t2
    if pd.isna(td) or td.total_seconds() == 0:
        return np.nan
    return round(l / td.total_seconds() * 3.6, 2)  # km/h

def calculate_jump_speed(row):
    l = 17.97 + 50 - 250 / 4  # 50m-AP1間距離（m）
    t1, t2 = row.get("50m"), row.get("AP1")
    if pd.isna(t1) or pd.isna(t2):
        return np.nan
    td = t1 - t2
    if pd.isna(td) or td.total_seconds() == 0:
        return np.nan
    return round(l / td.total_seconds() * 3.6, 2)  # km/h

def calculate_time_000_to_100(row):
    t1, t2 = row.get("150m"), row.get("50m")
    if pd.isna(t1) or pd.isna(t2):
        return np.nan
    td = t1 - t2
    if pd.isna(td) or td.total_seconds() == 0:
        return np.nan
    return round(td.total_seconds(), 2)

def calculate_time_100_to_200(row):
    t1, t2 = row.get("0m_2nd"), row.get("150m")
    if pd.isna(t1) or pd.isna(t2):
        return np.nan
    td = t1 - t2
    if pd.isna(td) or td.total_seconds() == 0:
        return np.nan
    return round(td.total_seconds(), 2)

def calculate_time_000_to_200(row):
    t1, t2 = row.get("0m_2nd"), row.get("50m")
    if pd.isna(t1) or pd.isna(t2):
        return np.nan
    td = t1 - t2
    if pd.isna(td) or td.total_seconds() == 0:
        return np.nan
    return round(td.total_seconds(), 2)

# utils.py 内：KPI計算の直後あたりに追加
def _propagate_imputed_flags_to_kpi(df: pd.DataFrame) -> pd.DataFrame:
    """
    タイム列の imputed__* を KPI列へ伝搬する。
    df に存在する列だけで安全に OR を取る。
    """
    # ← 添付 utils.py の計算式に合わせた依存関係
    deps_map = {
        "entry_speed":   ["FP_start", "0m_start"],
        "jump_speed":    ["50m", "AP1"],
        "Time000to100":  ["150m", "50m"],
        "Time100to200": ["0m_2nd", "150m"],
        "Time000to200":  ["0m_2nd", "50m"],
    }

    for kpi, deps in deps_map.items():
        if kpi not in df.columns:
            continue
        m = pd.Series(False, index=df.index)
        for d in deps:
            fcol = f"imputed__{d}"
            if fcol in df.columns:
                m = m | df[fcol].fillna(False)
        df[f"imputed__{kpi}"] = m
    return df

def fetch_df_from_db(query):
    DESIRED_ORDER = [
    "first_name", "last_name", "Date",
    "entry_speed", "jump_speed",
    "Time000to100", "Time100to200", "Time000to200",
    "FP_start", "0m_start", "60m", "AP1", "50m", "100m",
    "BP", "150m", "AP2", "200m", "FP_2nd", "0m_2nd"
    ]
    df = get_df_from_db(query)
    df["position"] = df["decoder_id"].map(translate_dict).fillna("Unknown")
    users = df["first_name"].unique()
    all_dfs = []

    for _, group in df.groupby("user_id"):
        group = group.sort_values(by=["timestamp"])
        temp = split_laps(group)
        temp = impute_times_by_distance(temp) # 保管処理
        temp["entry_speed"] = temp.apply(calculate_entry_speed, axis=1)
        temp["jump_speed"] = temp.apply(calculate_jump_speed, axis=1)
        temp["Time000to100"] = temp.apply(calculate_time_000_to_100, axis=1)
        temp["Time100to200"] = temp.apply(calculate_time_100_to_200, axis=1)
        temp["Time000to200"] = temp.apply(calculate_time_000_to_200, axis=1)
        
        # 補間フラグ
        temp = _propagate_imputed_flags_to_kpi(temp)
        temp["first_name"] = group["first_name"].iloc[0]
        temp["last_name"] = group["last_name"].iloc[0]
        imputed_cols = [c for c in temp.columns if c.startswith("imputed__")]
        ordered_cols = DESIRED_ORDER + [c for c in imputed_cols if c not in DESIRED_ORDER]
        temp = temp[ordered_cols]
        
        all_dfs.append(temp)

    # すべてを1つの DataFrame にまとめる
    result_df = pd.concat(all_dfs, ignore_index=True)
    return result_df,users

    """
    入力: 通過イベントの縦持ちDF（ユーザー・時系列・position 必須）
    出力: 欠損地点を仮想行で補完した縦持ちDF（is_imputed 付き）
    """
    # 想定: df に position 列がある（decoder_id→position は事前に付与）
    use_key = "user_id" if "user_id" in df.columns else "transponder_id"
    df = df.sort_values(["first_name","last_name","timestamp"]).copy()
    out_list = []
    for _, g in df.groupby(use_key, dropna=False):
        g = g.sort_values("timestamp")
        g = _assign_lap_id(g)
        for _, lap in g.groupby("lap_id"):
            imputed = _impute_one_lap(lap)
            if not imputed.empty:
                out_list.append(imputed)
    if not out_list:
        # 何もできなければ空
        return df.assign(is_imputed=False)
    out = pd.concat(out_list, ignore_index=True)
    out = out.sort_values(["first_name","last_name","timestamp"]).reset_index(drop=True)
    return out