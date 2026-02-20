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

def get_df_from_db(query):
    engine = create_engine(_get_db_url_from_settings(), pool_pre_ping=True)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


import unicodedata, re

def split_laps(df, all_data=None, log_file=None):
    """
    シンプルなラップ分割:
      - データは「FP,0m,60m,AP1,50m,100m,BP,150m,AP2,200m」の順で並ぶことを前提
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
    
    # 期待される順序（FPから200mまで）
    expected_order = ["FP", "0m", "60m", "AP1", "50m", "100m", "BP", "150m", "AP2", "200m"]
    
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
    
    # シンプルな分割：すべてのデータを拾う
    rows = []
    current_lap = {}
    
    logger.info(f"データ処理開始: {len(df)}行")
    
    def complete_lap(lap_dict, lap_id):
        """ラップを完成させてrowsに追加する（タイムスタンプ順に並べ替え）"""
        # SB1とFP列を0mの左に配置
        # 0mより5秒以内のSB1があればSB1列に入れる、0mより5秒以内のFPがあればFP列に入れる
        sb1_value = None
        fp_value = None
        
        if "0m" in lap_dict and lap_dict["0m"] is not None:
            zero_m_time = pd.to_datetime(lap_dict["0m"])
            time_window_start = zero_m_time - timedelta(seconds=5)
            time_window_end = zero_m_time
            
            # 0mより5秒以内のSB1を検索
            if sb1_candidates is not None and len(sb1_candidates) > 0:
                ts_col = pd.to_datetime(sb1_candidates["timestamp"])
                sb1_in_range = sb1_candidates[
                    (ts_col >= time_window_start) &
                    (ts_col <= time_window_end)
                ]
                
                if len(sb1_in_range) > 0:
                    # 最も近いSB1を採用（0mに最も近いもの）
                    sb1_in_range = sb1_in_range.sort_values("timestamp", ascending=False)
                    sb1_value = sb1_in_range.iloc[0]["timestamp"]
                    logger.info(f"LapID {lap_id}: SB1を検出 {sb1_value} (0m: {zero_m_time})")
            
            # 0mより5秒以内のFPを検索（df内から検索）
            fp_candidates = df[df["position"] == "FP"].copy()
            if len(fp_candidates) > 0:
                ts_col = pd.to_datetime(fp_candidates["timestamp"])
                fp_in_range = fp_candidates[
                    (ts_col >= time_window_start) &
                    (ts_col <= time_window_end)
                ]
                
                if len(fp_in_range) > 0:
                    # 最も近いFPを採用（0mに最も近いもの）
                    fp_in_range = fp_in_range.sort_values("timestamp", ascending=False)
                    fp_value = fp_in_range.iloc[0]["timestamp"]
                    logger.info(f"LapID {lap_id}: FPを検出 {fp_value} (0m: {zero_m_time})")
        
        # タイムスタンプ順に並べ替え
        sorted_lap = {}
        # タイムスタンプでソート
        sorted_items = sorted(lap_dict.items(), key=lambda x: x[1] if x[1] is not None else pd.Timestamp.max)
        for pos, ts in sorted_items:
            # FPは後で設定するので、ここではスキップ
            if pos != "FP":
                sorted_lap[pos] = ts
        
        # SB1とFPを追加（0mの左に配置するため、後で列順序を調整）
        sorted_lap["SB1"] = sb1_value
        sorted_lap["FP"] = fp_value
        
        # LapIDを追加
        sorted_lap["LapID"] = lap_id
        
        # セットを追加
        rows.append(sorted_lap.copy())
        logger.info(f"LapID {lap_id} 完了: {sorted_lap}")
    
    # 0mを検出したら新しいLapIDを割り振る
    # 時系列順に来たデータを順に格納していく
    lap_id = 0
    for idx, row in df.iterrows():
        pos = row["position"]
        ts = row["timestamp"]
        
        # 0mを検出したら新しいラップとして開始
        if pos == "0m":
            # 前のラップがあれば完成させる
            if current_lap:
                lap_id += 1
                complete_lap(current_lap, lap_id)
                current_lap = {}
            
            # 新しいラップを開始
            current_lap[pos] = ts
        else:
            # その他の位置は現在のラップに追加（current_lapが空でない場合のみ）
            # 同じ位置が複数回来た場合は、最新のタイムスタンプで上書き
            if current_lap:
                current_lap[pos] = ts
    
    # 最後のラップを完成させる
    if current_lap:
        lap_id += 1
        complete_lap(current_lap, lap_id)
    
    logger.info(f"処理完了: {len(rows)}個のラップを取得（一部空欄を含む可能性あり）")
    
    if not rows:
        logger.warning("ラップが0件でした")
        columns_with_lapid = ["LapID", "SB1", "FP"] + expected_order
        return pd.DataFrame(columns=columns_with_lapid)
    
    # DataFrameに変換（列順序は統一、各行のデータはタイムスタンプ順）
    # すべての列を収集
    all_columns = set(["LapID", "SB1", "FP"])
    for lap_dict in rows:
        all_columns.update(lap_dict.keys())
    # LapIDを最初に、その後はSB1、FP、0mの順で、その後はexpected_orderの残り、最後にその他の列
    ordered_columns = ["LapID", "SB1", "FP"]
    # expected_orderからFPを除外して追加（FPは既に追加済み）
    for col in expected_order:
        if col in all_columns and col != "FP":
            ordered_columns.append(col)
    for col in sorted(all_columns - set(ordered_columns)):
        if col not in ["LapID", "SB1", "FP"]:
            ordered_columns.append(col)
    
    result_df = pd.DataFrame(rows, columns=ordered_columns)
    
    # Date列を追加（最初のタイムスタンプの日付、0mがNoneの場合は最初の有効なタイムスタンプを使用）
    if len(expected_order) > 0 and expected_order[0] in result_df.columns:
        first_col = expected_order[0]
        # 0mがNoneの場合は、最初の有効なタイムスタンプを探す
        def get_date(row):
            # SB1があればSB1を使用、SB1がなくてFPがあればFPを使用、なければ0mから順に有効なタイムスタンプを探す
            if "SB1" in row and pd.notna(row["SB1"]) and row["SB1"] is not None:
                return pd.to_datetime(row["SB1"]).date()
            if "FP" in row and pd.notna(row["FP"]) and row["FP"] is not None:
                return pd.to_datetime(row["FP"]).date()
            for col in expected_order:
                if col in row and pd.notna(row[col]) and row[col] is not None:
                    return pd.to_datetime(row[col]).date()
            return None
        result_df.insert(0, "Date", result_df.apply(get_date, axis=1))
    
    logger.info(f"結果DataFrame: {len(result_df)}行 x {len(result_df.columns)}列")
    logger.info(f"列名: {result_df.columns.tolist()}")
    if "SB1" in result_df.columns:
        logger.info(f"SB1非空値: {result_df['SB1'].notna().sum()}件")
    if "FP" in result_df.columns:
        logger.info(f"FP非空値: {result_df['FP'].notna().sum()}件")
    
    return result_df


def fetch_df_from_db(
    query: str,
    progress: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
):
    """
    DB→前処理→split→結合 を行う。
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
    if "FP" in result_df.columns:
        print(f"[DBG] result FP_nonnull: {int(result_df['FP'].notna().sum())}")

    _p("完了")
    return result_df, users
