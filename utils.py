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

def split_laps(df, all_data=None, log_file=None):
    """
    シンプルなラップ分割:
      - データは「0m,60m,AP1,50m,100m,BP,150m,AP2,200m,FP」の順で並ぶことを前提
      - この順序で1セット（1ラップ）として扱う
      - 完全にそろっているセットだけを1行目、2行目として表示する
      - SB1列を追加（0mの左に配置、user_idが紐づいていないデータから0mから5秒前までを検索）
      - 処理結果はログファイルに記録
    """
    import logging
    from datetime import timedelta
    
    # ログファイルの設定
    if log_file is None:
        log_file = os.path.join(os.getcwd(), "lap_processing.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # 期待される順序
    expected_order = ["0m", "60m", "AP1", "50m", "100m", "BP", "150m", "AP2", "200m", "FP"]
    
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    
    # position列の正規化
    def _canon_pos(x: str) -> str:
        if x is None:
            return ""
        s = unicodedata.normalize("NFKC", str(x)).strip()
        u = s.upper()
        if u in {"FP", "ＦＰ"}:
            return "FP"
        if u in {"0M", "OM", "0Ｍ"}:
            return "0m"
        if u in {"SB1", "SB-1", "SB 1", "ＳＢ１"}:
            return "SB1"
        return s
    
    df["position"] = df["position"].map(_canon_pos)
    
    # 全データからSB1を検索するための準備（user_idが紐づいていないSB1データ）
    sb1_candidates = None
    if all_data is not None:
        all_data = all_data.copy()
        all_data["timestamp"] = pd.to_datetime(all_data["timestamp"], errors="coerce")
        all_data["position"] = all_data["position"].map(_canon_pos)
        # user_idが紐づいていない（NoneまたはNaN）かつSB1のデータを取得
        if "user_id" in all_data.columns:
            sb1_candidates = all_data[
                (all_data["position"] == "SB1") & 
                (all_data["user_id"].isna() | (all_data["user_id"] == ""))
            ].copy()
            logger.info(f"SB1候補データ: {len(sb1_candidates)}件")
    
    # 順序通りに分割
    rows = []
    current_lap = {}
    expected_idx = 0
    
    logger.info(f"データ処理開始: {len(df)}行")
    logger.info(f"期待される順序: {expected_order}")
    
    def complete_lap(lap_dict, lap_num):
        """ラップを完成させてrowsに追加する"""
        # SB1を検索（0mから5秒前まで）
        sb1_value = None
        if sb1_candidates is not None and "0m" in lap_dict and lap_dict["0m"] is not None:
            zero_m_time = lap_dict["0m"]
            time_window_start = zero_m_time - timedelta(seconds=5)
            time_window_end = zero_m_time
            
            # 0mから5秒前までの範囲でSB1を検索
            sb1_in_range = sb1_candidates[
                (sb1_candidates["timestamp"] >= time_window_start) &
                (sb1_candidates["timestamp"] <= time_window_end)
            ]
            
            if len(sb1_in_range) > 0:
                # 最も近いSB1を採用（0mに最も近いもの）
                sb1_in_range = sb1_in_range.sort_values("timestamp", ascending=False)
                sb1_value = sb1_in_range.iloc[0]["timestamp"]
                logger.info(f"ラップ {lap_num}: SB1を検出 {sb1_value} (0m: {zero_m_time})")
        
        # SB1を追加
        lap_dict["SB1"] = sb1_value
        
        # セットを追加
        rows.append(lap_dict.copy())
        logger.info(f"ラップ {lap_num} 完了: {lap_dict}")
    
    for idx, row in df.iterrows():
        pos = row["position"]
        ts = row["timestamp"]
        
        # 期待される順序の位置を確認
        if pos in expected_order:
            # 既にラップが完成している場合（expected_idx >= len(expected_order)）
            if expected_idx >= len(expected_order):
                # 前のラップが不完全な場合の処理（FPが抜けている）
                if current_lap:
                    # FPが抜けている：FPを空欄にして完成
                    current_lap["FP"] = None
                    logger.info(f"FPが抜けているため空欄に設定してラップを完成")
                    complete_lap(current_lap, len(rows) + 1)
                    current_lap = {}
                
                # 新しいラップを開始
                if pos == expected_order[0]:  # 0mから開始
                    current_lap[pos] = ts
                    expected_idx = 1
                continue
            
            expected_pos = expected_order[expected_idx]
            
            if pos == expected_pos:
                # 期待通りの位置
                current_lap[pos] = ts
                expected_idx += 1
                
                # 1セット完了（FPまで到達）
                if expected_idx >= len(expected_order):
                    complete_lap(current_lap, len(rows) + 1)
                    current_lap = {}
                    expected_idx = 0
            else:
                # 順序が合わない場合、何個抜けているかチェック
                # 循環を考慮：FPの後は0mが来る
                found_match = False
                
                # 200mまで到達した後、次のデータが0m（新しいラップの開始）の場合、FPが抜けている
                if expected_idx == len(expected_order) - 1 and pos == expected_order[0]:
                    # FPが抜けている：FPを空欄にしてラップを完成
                    current_lap["FP"] = None
                    logger.info(f"FPが抜けているため空欄に設定してラップを完成")
                    complete_lap(current_lap, len(rows) + 1)
                    current_lap = {}
                    
                    # 新しいラップを開始
                    current_lap[pos] = ts
                    expected_idx = 1
                    found_match = True
                else:
                    # 1つ先、2つ先...とチェック（最大2つまで）
                    # 循環を考慮：FPの後は0mが来るので、check_idxが範囲外の場合は0mをチェック
                    for skip_count in range(1, min(3, len(expected_order) - expected_idx + 2)):
                        check_idx = expected_idx + skip_count
                        
                        # 循環を考慮：範囲外の場合は0m（expected_order[0]）をチェック
                        if check_idx < len(expected_order):
                            check_pos = expected_order[check_idx]
                        elif check_idx == len(expected_order) and pos == expected_order[0]:
                            # FPの後は0mが来る（循環）
                            check_pos = expected_order[0]
                        else:
                            continue
                        
                        if pos == check_pos:
                            # skip_count個抜けている
                            if check_idx < len(expected_order):
                                missing_positions = [expected_order[i] for i in range(expected_idx, check_idx)]
                            else:
                                # FPが抜けている場合
                                missing_positions = [expected_order[expected_idx]]
                            
                            if skip_count == 1:
                                # 1つだけ抜けている：空欄にして続行
                                current_lap[missing_positions[0]] = None
                                logger.info(f"位置 {missing_positions[0]} が抜けているため空欄に設定。次の位置 {pos} を記録")
                                current_lap[pos] = ts
                                
                                if check_idx < len(expected_order):
                                    expected_idx = check_idx + 1
                                else:
                                    # FPの後は0m（循環）
                                    expected_idx = 1
                                
                                # 1セット完了（FPまで到達）
                                if expected_idx >= len(expected_order):
                                    complete_lap(current_lap, len(rows) + 1)
                                    current_lap = {}
                                    expected_idx = 0
                                found_match = True
                                break
                            else:
                                # 2つ以上抜けている：ラップを廃棄
                                logger.warning(f"不完全なラップを破棄（{skip_count}個の位置が抜けている: {missing_positions}）: {current_lap}")
                                current_lap = {}
                                expected_idx = 0
                                
                                # 新しいセットを開始
                                if pos == expected_order[0]:  # 0mから開始
                                    current_lap[pos] = ts
                                    expected_idx = 1
                                found_match = True
                                break
                
                if not found_match:
                    # どの期待位置とも一致しない場合、ラップを廃棄
                    if current_lap:
                        logger.warning(f"順序不一致でラップ破棄: 期待={expected_pos}, 実際={pos}, 現在のセット={current_lap}")
                    current_lap = {}
                    expected_idx = 0
                    
                    # 新しいセットを開始
                    if pos == expected_order[0]:  # 0mから開始
                        current_lap[pos] = ts
                        expected_idx = 1
    
    # 最後のセットが不完全な場合、1つだけ抜けている場合は完成させる
    if current_lap and expected_idx < len(expected_order):
        missing_count = len(expected_order) - expected_idx
        if missing_count == 1:
            # 1つだけ抜けている場合、最後の位置を空欄にして完成
            last_expected_pos = expected_order[expected_idx]
            current_lap[last_expected_pos] = None
            logger.info(f"最後の位置 {last_expected_pos} が抜けているため空欄に設定してラップを完成")
            complete_lap(current_lap, len(rows) + 1)
        else:
            # 2つ以上抜けている場合は破棄
            missing_positions = [expected_order[i] for i in range(expected_idx, len(expected_order))]
            logger.warning(f"不完全なラップを破棄（{missing_count}個の位置が抜けている: {missing_positions}）: {current_lap}")
    
    logger.info(f"処理完了: {len(rows)}個のラップを取得（一部空欄を含む可能性あり）")
    
    if not rows:
        logger.warning("ラップが0件でした")
        columns_with_sb1 = ["SB1"] + expected_order
        return pd.DataFrame(columns=columns_with_sb1)
    
    # DataFrameに変換（SB1列を含む）
    columns_with_sb1 = ["SB1"] + expected_order
    result_df = pd.DataFrame(rows, columns=columns_with_sb1)
    
    # Date列を追加（最初のタイムスタンプの日付、0mがNoneの場合は最初の有効なタイムスタンプを使用）
    if len(expected_order) > 0 and expected_order[0] in result_df.columns:
        first_col = expected_order[0]
        # 0mがNoneの場合は、最初の有効なタイムスタンプを探す
        def get_date(row):
            # 0mから順に有効なタイムスタンプを探す
            for col in expected_order:
                if col in row and pd.notna(row[col]) and row[col] is not None:
                    return pd.to_datetime(row[col]).date()
            return None
        result_df.insert(0, "Date", result_df.apply(get_date, axis=1))
    
    # SB1列を0mの左に移動（Date列の後、0mの前）
    if "SB1" in result_df.columns and expected_order[0] in result_df.columns:
        # SB1列を一旦削除してから0mの前に挿入
        sb1_col = result_df.pop("SB1")
        zero_m_idx = result_df.columns.get_loc(expected_order[0])
        result_df.insert(zero_m_idx, "SB1", sb1_col)
    
    logger.info(f"結果DataFrame: {len(result_df)}行 x {len(result_df.columns)}列")
    logger.info(f"列名: {result_df.columns.tolist()}")
    logger.info(f"SB1非空値: {result_df['SB1'].notna().sum()}件")
    
    return result_df


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
    DB→前処理→split→補間→結合 を行う。
    - progress(msg): 進捗メッセージをUIへ通知（未指定なら無視）
    - cancel_event: .set() されていたら安全に中断（未指定なら無視）
    戻り値: (result_df, users)
    """

    def _p(msg: str):
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

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
        # スキーマだけ維持した空DFを返す
        return df.copy(), users

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
    
    # デバッグ: 位置ラベルの内訳
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
    
    # ログファイルのパスを設定
    log_file = os.path.join(os.getcwd(), "lap_processing.log")

    for uid, group in groups:
        _check_cancel()
        _p(f"ユーザー {uid if uid is not None else '-'}: split中…")

        # タイムスタンプでソート
        group = group.sort_values(by=["timestamp"]).reset_index(drop=True)

        # --- split（シンプルな順序ベースの分割、全データを渡してSB1検索用に使用）---
        temp = split_laps(group, all_data=df, log_file=log_file)

        # --- 氏名付与 ---
        if "first_name" in group.columns:
            temp["first_name"] = group["first_name"].iloc[0]
        if "last_name" in group.columns:
            temp["last_name"] = group["last_name"].iloc[0]
        if "user_id" in group.columns:
            temp["user_id"] = uid

        # 列整形はここでは行わず、そのまま使う
        all_dfs.append(temp.copy())

    # 5) 結合（空安全）
    if not all_dfs:
        _p("集計結果0件")
        # 元DFのスキーマだけ持った空DF
        return df.iloc[0:0].copy(), users

    _p("結合中…")
    print(f"[DBG] concat about to merge {len(all_dfs)} dfs")
    result_df = pd.concat(all_dfs, ignore_index=True)
    print(f"[DBG] result columns: {result_df.columns.tolist()}")
    if "SB1" in result_df.columns:
        print(f"[DBG] result SB1_nonnull: {int(result_df['SB1'].notna().sum())}")

    _p("完了")
    return result_df, users
