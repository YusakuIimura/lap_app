import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Callable, Optional
import threading

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

SETTINGS_PATH = os.path.join(os.getcwd(), "settings.json")

def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _get_db_url_from_settings():
    s = _load_json(SETTINGS_PATH)
    db = s.get("db", {})
    if "url" in db and db["url"]:
        return db["url"]
    driver = db.get("driver", "mysql+pymysql")
    host   = db.get("host", "127.0.0.1")
    port   = int(db.get("port", 3306))
    name   = db.get("name")  # DB名
    user   = db.get("user")
    passwd = db.get("pass")
    if not all([name, user, passwd]):
        raise RuntimeError("setting.json の db.name / db.user / db.pass を設定してください。")
    return f"{driver}://{user}:{passwd}@{host}:{port}/{name}"

def _noop(*args, **kwargs): pass

def jst_str_to_utc_sql(ts_jst_str: str) -> str:
    """
    'YYYY-MM-DD HH:MM:SS' (JST) を UTC の同フォーマット文字列に変換。
    SQLの BETWEEN / 比較に使う。
    """
    if not ts_jst_str:
        return ts_jst_str
    dt = datetime.strptime(ts_jst_str, "%Y-%m-%d %H:%M:%S")
    jst = dt.replace(tzinfo=ZoneInfo("Asia/Tokyo"))
    utc = jst.astimezone(ZoneInfo("UTC"))
    return utc.strftime("%Y-%m-%d %H:%M:%S")

def to_jst_naive(series: pd.Series) -> pd.Series:
    """
    Series内の日時を UTC として解釈 → JSTへ変換 → tzを外したnaiveに。
    DBからUTCで来たtimestamp列に適用する想定。
    """
    s = pd.to_datetime(series, errors="coerce", utc=True)
    s = s.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return s

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

def get_df_from_db(query):
    engine = create_engine(_get_db_url_from_settings(), pool_pre_ping=True)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


import unicodedata, re

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
      - SB1: FP_start～0m_start の間に存在すればその時刻を1回だけ記録（補間なし）
    """
    def _canon_pos(x: str) -> str:
        if x is None:
            return ""
        s = unicodedata.normalize("NFKC", str(x)).strip()
        u = s.upper()
        if u in {"FP", "ＦＰ"}:
            return "FP"
        if u in {"0M", "OM", "0Ｍ"}:
            return "0m"
        # SB1 バリアント（SB-1, SB 1, ＳＢ１ など）
        if re.fullmatch(r"SB[\s\-]*1", u) or u == "ＳＢ１":
            return "SB1"
        return s  # 既知以外はそのまま（NFKC+trim 済み）

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # 正規化した position 列を用意（以降はこれだけ使う）
    df["position"] = df["position"].map(_canon_pos)

    anchors   = {"FP", "0m"}
    inner_pts = ["60m", "AP1", "50m", "100m", "BP", "150m", "AP2", "200m"]

    tmp_cols = ["FP_first", "SB1", "0m_start"] + inner_pts
    rows = []
    cur = {k: None for k in tmp_cols}
    anchor = None  # 現在のラップのアンカー: 'FP' or '0m' or None

    def flush():
        nonlocal cur, rows, anchor
        if any(v is not None for v in cur.values()):
            rows.append(cur)
        cur = {k: None for k in tmp_cols}
        anchor = None

    for _, r in df.iterrows():
        pos = r["position"]
        ts  = r["timestamp"]

        # 想定外の地点は無視（SB1 は許可）
        if (pos not in anchors) and (pos not in inner_pts) and (pos != "SB1"):
            continue

        # --- アンカー処理（FP / 0m）---
        if pos in anchors:
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
                    # 異なるアンカーが途中で出た → 値だけ記録し継続
                    if pos == "FP" and cur["FP_first"] is None:
                        cur["FP_first"] = ts
                    if pos == "0m" and cur["0m_start"] is None:
                        cur["0m_start"] = ts
            continue

        # --- SB1（FP～0mの間に1回だけ）---
        if pos == "SB1":
            fp_seen     = (cur.get("FP_first") is not None) and (not pd.isna(cur.get("FP_first")))
            z0_not_seen = (cur.get("0m_start") is None) or pd.isna(cur.get("0m_start"))
            sb1_not_set = (cur.get("SB1") is None) or pd.isna(cur.get("SB1"))
            if (anchor is not None) and fp_seen and z0_not_seen and sb1_not_set:
                cur["SB1"] = ts
                print(f"[DBG split] SB1 recorded at {ts}")
            continue

        # --- 内側ポイント（60m, AP1, ...）：最初の観測だけ記録 ---
        if anchor is not None and pos in inner_pts and cur[pos] is None:
            cur[pos] = ts

    # 末尾も確定
    flush()

    if not rows:
        return pd.DataFrame(columns=["FP_start", "SB1", "0m_start", *inner_pts, "FP_2nd", "0m_2nd"])

    out_tmp = pd.DataFrame(rows, columns=tmp_cols)

    # 表示/計算用の FP_start を作る（0mより後のFPは欠測扱いにして負のentry_speedを防止）
    def compute_fp_start(row):
        fp = row["FP_first"]
        z0 = row["0m_start"]
        if pd.isna(fp):
            return pd.NaT
        if pd.isna(z0):
            return fp
        return fp if fp <= z0 else pd.NaT

    out = pd.DataFrame()
    out["FP_start"] = out_tmp.apply(compute_fp_start, axis=1)
    out["SB1"]      = out_tmp["SB1"]                      # 補間なし（実測のみ）
    out["0m_start"] = out_tmp["0m_start"]
    for c in inner_pts:
        out[c] = out_tmp[c]

    # _2nd は「次行の start」をシフトで付与
    out["FP_2nd"] = out_tmp["FP_first"].shift(-1)   # 次ラップの最初のFP（生値）
    out["0m_2nd"] = out["0m_start"].shift(-1)

    # 年月日列（ラップ内で最初に観測できた時刻の日付）
    cols_for_date = ["FP_first", "0m_start"] + inner_pts
    first_ts = out_tmp[cols_for_date].apply(
        lambda r: r.dropna().min() if r.notna().any() else pd.NaT, axis=1
    )
    out.insert(0, "Date", pd.to_datetime(first_ts).dt.date)

    try:
        if "SB1" in out.columns:
            print(f"[DBG][split_laps] SB1_nonnull: {int(out['SB1'].notna().sum())}")
        else:
            print("[DBG][split_laps] SB1 column missing in output")
    except Exception as e:
        print(f"[DBG][split_laps] SB1 check failed: {e}")

    return out


def calculate_entry_speed(row):
    # l = 17.97  # 0m-FP間距離（m）
    # t1, t2 = row.get("0m_start"), row.get("FP_start")
    # if pd.isna(t1) or pd.isna(t2):
    #     return np.nan
    # td = t1 - t2
    # if pd.isna(td) or td.total_seconds() == 0:
    #     return np.nan
    # return round(l / td.total_seconds() * 3.6, 2)  # km/h
    return ''

def calculate_jump_speed(row):
    # l = 17.97 + 50 - 250 / 4  # 50m-AP1間距離（m）
    # t1, t2 = row.get("50m"), row.get("AP1")
    # if pd.isna(t1) or pd.isna(t2):
    #     return np.nan
    # td = t1 - t2
    # if pd.isna(td) or td.total_seconds() == 0:
    #     return np.nan
    # return round(l / td.total_seconds() * 3.6, 2)  # km/h
    return ''

def calculate_time_000_to_100(row):
    t1, t2 = row.get("150m"), row.get("50m")
    if pd.isna(t1) or pd.isna(t2):
        return np.nan
    td = t1 - t2
    if pd.isna(td) or td.total_seconds() == 0:
        return np.nan
    return round(td.total_seconds(), 3)

def calculate_time_100_to_200(row):
    t1, t2 = row.get("0m_2nd"), row.get("150m")
    if pd.isna(t1) or pd.isna(t2):
        return np.nan
    td = t1 - t2
    if pd.isna(td) or td.total_seconds() == 0:
        return np.nan
    return round(td.total_seconds(), 3)

def calculate_time_000_to_200(row):
    t1, t2 = row.get("0m_2nd"), row.get("50m")
    if pd.isna(t1) or pd.isna(t2):
        return np.nan
    td = t1 - t2
    if pd.isna(td) or td.total_seconds() == 0:
        return np.nan
    return round(td.total_seconds(), 3)

def calculate_time_000_to_625(row):
    t1, t2 = row.get("AP1"), row.get("FP_start")
    if pd.isna(t1) or pd.isna(t2):
        return np.nan
    td = t1 - t2
    if pd.isna(td) or td.total_seconds() == 0:
        return np.nan
    return round(td.total_seconds(), 3)

def calculate_time_625_to_125(row):
    t1, t2 = row.get("BP"), row.get("AP1")
    if pd.isna(t1) or pd.isna(t2):
        return np.nan
    td = t1 - t2
    if pd.isna(td) or td.total_seconds() == 0:
        return np.nan
    return round(td.total_seconds(), 3)

def calculate_time_000_to_125(row):
    t1, t2 = row.get("BP"), row.get("FP_start")
    if pd.isna(t1) or pd.isna(t2):
        return np.nan
    td = t1 - t2
    if pd.isna(td) or td.total_seconds() == 0:
        return np.nan
    return round(td.total_seconds(), 3)

def calculate_time_125_to_250(row):
    """BP → FP_2nd の秒数。どちらか欠けていたら NaN。0以下は NaN。"""
    t0, t1 = row.get("BP"), row.get("FP_2nd")
    if pd.isna(t0) or pd.isna(t1):
        return np.nan
    dt = t1 - t0
    return round(dt.total_seconds(), 3) if pd.notna(dt) and dt.total_seconds() > 0 else np.nan

def calculate_time_000_to_625_from_sb(row):
    """standing 用：SB1→AP1 の秒数（SB1が無ければ NaN、補完しない）"""
    t0, t1 = row.get("SB1"), row.get("AP1")
    if pd.isna(t0) or pd.isna(t1):
        return np.nan
    td = t1 - t0
    if pd.isna(td) or td.total_seconds() <= 0:
        return np.nan
    return round(td.total_seconds(), 3)

def calculate_time_000_to_125_from_sb(row):
    """standing 用：SB1→BP の秒数（SB1が無ければ NaN、補完しない）"""
    t0, t1 = row.get("SB1"), row.get("BP")
    if pd.isna(t0) or pd.isna(t1):
        return np.nan
    td = t1 - t0
    if pd.isna(td) or td.total_seconds() <= 0:
        return np.nan
    return round(td.total_seconds(), 3)


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

def fetch_df_from_db(
    query: str,
    progress: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
):
    """
    DB→前処理→split→補間→KPI→結合 を行う。
    - progress(msg): 進捗メッセージをUIへ通知（未指定なら無視）
    - cancel_event: .set() されていたら安全に中断（未指定なら無視）
    戻り値: (result_df, users)
    """
    DESIRED_ORDER = [
        "first_name", "last_name", "Date",
        "Time000to100", "Time100to200", "Time000to200",
        "Time000to625","Time625to125",  "Time000to125_roll", "Time000to125_stand", "Time125to250",
        "FP_start","SB1", "0m_start", "60m", "AP1", "50m", "100m",
        "BP", "150m", "AP2", "200m", "FP_2nd", "0m_2nd",
        "entry_speed", "jump_speed",
    ]

    def _p(msg: str):
        if progress:
            try: progress(msg)
            except Exception: pass

    def _check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("CancelledByUser")

    # 1) DB取得
    _p("DB問い合わせ中…")
    df = get_df_from_db(query)
    _check_cancel()

    # データ0件なら安全に返す
    users = df["first_name"].unique() if "first_name" in df.columns else []
    if df.empty:
        _p("データ0件")
        # 期待カラムの空DF（補間フラグ込み）を返す
        imputed_bases = [
            "FP_start","0m_start","60m","AP1","50m","100m",
            "BP","150m","AP2","200m","FP_2nd","0m_2nd",
            "entry_speed","jump_speed","Time000to100","Time100to200","Time000to200"
        ]
        empty_cols = DESIRED_ORDER + [f"imputed__{c}" for c in imputed_bases]
        return pd.DataFrame(columns=empty_cols), users

    # 2) タイムスタンプをJST（naive）へ
    _p("タイムスタンプ変換中…")
    if "timestamp" in df.columns:
        df["timestamp"] = to_jst_naive(df["timestamp"])
    _check_cancel()

    # 3) デコーダ→地点名
    _p("位置ラベル生成中…")
    if "decoder_id" in df.columns:
        df["position"] = df["decoder_id"].map(translate_dict).fillna("Unknown")
    else:
        df["position"] = "Unknown"
    _check_cancel()
    
    # ★ 追加: 位置ラベルの内訳を確認
    try:
        cnt = df["position"].value_counts(dropna=False).to_dict()
        print(f"[DBG] position counts (before per-user filter): {cnt}")
    except Exception as e:
        print(f"[DBG] position counts failed: {e}")

    # 4) ユーザーごと処理
    _p("ユーザーごとに集計中…")
    all_dfs = []
    group_key = "user_id" if "user_id" in df.columns else None
    groups = df.groupby(group_key) if group_key else [(None, df)]

    for uid, group in groups:
        _check_cancel()
        _p(f"ユーザー {uid if uid is not None else '-'}: split中…")
        group = group.sort_values(by=["timestamp"]).reset_index(drop=True)
        
        # ▼ 追加: 全体の SB1 行を取り込み（user_id が空でもOK）
        sb1_all = df[df["position"] == "SB1"].copy()
        # タイムレンジで軽く絞る（任意）：このユーザーのデータ期間±余白に限定
        if not group.empty:
            tmin, tmax = group["timestamp"].min(), group["timestamp"].max()
            sb1_all = sb1_all[(sb1_all["timestamp"] >= tmin) & (sb1_all["timestamp"] <= tmax)]
            
        # このユーザーの FP/0m は user_id==uid の行のみで定義
        is_fp_user = group["position"].eq("FP")
        is_0m_user = group["position"].eq("0m")

        # 区間ID（このユーザーのFPのみでカウント）
        fp_cum_user = is_fp_user.cumsum()
        z0_cum_user = is_0m_user.cumsum()

        # このユーザーの文脈で「直前FPがあり、まだ0mが来ていない」区間（ユーザー行のみ）
        between_user = fp_cum_user.gt(z0_cum_user)

        # ── ここから SB1 判定：SB1 は “全体から” 拾う ──
        # 各 FP 区間の「最初の 0m 時刻」をユーザー行から求めて map
        interval_id_user = fp_cum_user
        first0_ts_by_int = (
            group.assign(_is0m=is_0m_user)
                .groupby(interval_id_user, dropna=False)
                .apply(lambda s: s.loc[s["_is0m"], "timestamp"].iloc[0] if s["_is0m"].any() else pd.NaT)
        )

        # SB1 にも「どのユーザー区間に属するか」を割り当てる必要があるため、
        # SB1 をユーザー行と縦結合して “同一の時系列” 上で区間IDを引く。
        work = pd.concat([group, sb1_all], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

        # ユーザーの FP/0m のみで区間IDを進める（非ユーザー行は0のまま進行しないので forward-fill）
        is_fp_in_work = (work["position"].eq("FP")) & (work.get("user_id") == uid)
        is_0m_in_work = (work["position"].eq("0m")) & (work.get("user_id") == uid)

        fp_cum_work = is_fp_in_work.cumsum()
        z0_cum_work = is_0m_in_work.cumsum()

        # FP〜0m の “間” の判定を work 上で作る（ユーザー文脈）
        between_work = fp_cum_work.gt(z0_cum_work)

        # 各行の区間ID
        interval_id_work = fp_cum_work

        # 各行に「この区間の最初の0m時刻」を付与（ユーザー行から作ったテーブルを map）
        work["_first0"] = interval_id_work.map(first0_ts_by_int)

        # 条件を満たす SB1（SB1 ∧ ユーザー文脈で間にある ∧ 最初の0mより前）
        sb1_between = (work["position"].eq("SB1")) & between_work & work["timestamp"].lt(work["_first0"])

        # さらに “ちょうど1回だけ” 条件：区間ごとに SB1 件数を数えて1のものだけ
        sb1_cnt = sb1_between.groupby(interval_id_work, dropna=False).transform("sum")
        sb1_keep = sb1_between & sb1_cnt.eq(1)

        # このユーザーの最終入力：元のユーザー行 + 残すべき SB1 行
        # （ユーザー行は常に残す。SB1は keep のものだけ残す。他の非ユーザー行は捨てる）
        keep_mask = (work.get("user_id") == uid) | sb1_keep
        group_plus_sb1 = work[keep_mask].copy().sort_values("timestamp").reset_index(drop=True)

        # ▼ デバッグ（任意）
        try:
            pre = int((group["position"] == "SB1").sum())
            aft = int((group_plus_sb1["position"] == "SB1").sum())
            print(f"[DBG][uid={uid}] SB1 before={pre}, after-merge={aft}")
        except Exception:
            pass

    
        # --- split ---
        temp = split_laps(group_plus_sb1)
        
        # ★ 追加: split後、SB1列があるか＆非NaT件数
        cols = list(temp.columns)
        sb1_nonnull = int(temp["SB1"].notna().sum()) if "SB1" in temp.columns else "N/A"
        print(f"[DBG][uid={uid}] split cols: {cols} | SB1_nonnull: {sb1_nonnull}")

        # --- 補間 ---
        _p(f"ユーザー {uid if uid is not None else '-'}: 補間中…")
        temp = impute_times_by_distance(temp)
        temp = temp.reset_index(drop=True)

        # --- 2nd 再計算 ---
        temp["0m_2nd"] = temp["0m_start"].shift(-1)
        temp["FP_2nd"] = temp["FP_first"].shift(-1) if "FP_first" in temp.columns else temp["FP_start"].shift(-1)

        # --- KPI計算 ---
        _p(f"ユーザー {uid if uid is not None else '-'}: KPI計算中…")
        temp["entry_speed"]   = temp.apply(calculate_entry_speed, axis=1)
        temp["jump_speed"]    = temp.apply(calculate_jump_speed, axis=1)
        temp["Time000to100"]  = temp.apply(calculate_time_000_to_100, axis=1)
        temp["Time100to200"]  = temp.apply(calculate_time_100_to_200, axis=1)
        temp["Time000to200"]  = temp.apply(calculate_time_000_to_200, axis=1)
        temp["Time000to625"]  = temp.apply(calculate_time_000_to_625, axis=1)
        temp["Time625to125"]  = temp.apply(calculate_time_625_to_125, axis=1)
        temp["Time000to125_roll"]  = temp.apply(calculate_time_000_to_125, axis=1)
        temp["Time000to125_stand"] = temp.apply(calculate_time_000_to_125_from_sb, axis=1)
        temp["Time125to250"] = temp.apply(calculate_time_125_to_250, axis=1)

        # --- 補間フラグ伝搬 & 氏名付与 ---
        temp = _propagate_imputed_flags_to_kpi(temp)
        if "first_name" in group.columns:
            temp["first_name"] = group["first_name"].iloc[0]
        if "last_name" in group.columns:
            temp["last_name"]  = group["last_name"].iloc[0]
        

        # 列整形（存在する imputed__* を末尾に）
        imputed_cols = [c for c in temp.columns if c.startswith("imputed__")]
        ordered_cols = [c for c in DESIRED_ORDER if c in temp.columns] + [c for c in imputed_cols if c not in DESIRED_ORDER]
        temp = temp.reindex(columns=ordered_cols)

        all_dfs.append(temp)

    # 5) 結合（空安全）
    if not all_dfs:
        _p("集計結果0件")
        imputed_bases = [
            "FP_start","0m_start","60m","AP1","50m","100m",
            "BP","150m","AP2","200m","FP_2nd","0m_2nd",
            "entry_speed","jump_speed","Time000to100","Time100to200","Time000to200"
        ]
        empty_cols = DESIRED_ORDER + [f"imputed__{c}" for c in imputed_bases]
        return pd.DataFrame(columns=empty_cols), users

    _p("結合中…")
    print(f"[DBG] concat about to merge {len(all_dfs)} dfs")
    result_df = pd.concat(all_dfs, ignore_index=True)
    print(f"[DBG] result columns: {result_df.columns.tolist()}")
    if "SB1" in result_df.columns:
        print(f"[DBG] result SB1_nonnull: {int(result_df['SB1'].notna().sum())}")

    _p("完了")
    return result_df, users
