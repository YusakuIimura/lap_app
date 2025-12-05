"""
KPI取得API - SIer向けシンプルインターフェース

このモジュールは、UIを実装する外部ベンダー（SIer）に提供するための
シンプルなAPIを提供します。

主な機能:
- 指定期間内の全選手のラップデータを自動取得
- 全KPI（Rolling/Standing/Flying）を計算
- 欠損値を距離比例で自動補間
- シンプルなリスト形式で返却
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from utils import fetch_df_from_db, jst_str_to_utc_sql
import csv

def get_all_kpis(start_time: str, end_time: str) -> Dict[str, List[List[Any]]]:
    """
    指定期間内の全選手の全ラップデータを取得

    この関数は、DBから所定の時間区間においてデータが存在するライダーを全て列挙し、
    そのライダーごとに、すべての登録されている時刻を列挙します。
    欠損値（NAN）は距離比例で自動補間されます。

    Parameters
    ----------
    start_time : str
        開始時刻 (JST形式) "YYYY-MM-DD HH:MM:SS"
        例: "2025-01-01 14:00:00"

    end_time : str
        終了時刻 (JST形式) "YYYY-MM-DD HH:MM:SS"
        例: "2025-01-01 18:00:00"

    Returns
    -------
    Dict[str, List[List[Any]]]
        選手名をキーとし、各選手のラップデータ（リストのリスト）を値とする辞書

        各ラップ（内側のリスト）の構成:
        [
            時刻(Date),
            # KPI - Rolling/Standing共通
            Time000to625,        # 0m → 62.5m (AP1)
            Time625to125,        # 62.5m (AP1) → 125m (BP)
            Time125to250,        # 125m (BP) → 250m (FP_2nd)

            # KPI - Rolling専用
            Time000to125_roll,   # 0m → 125m (Rolling start)

            # KPI - Standing専用
            Time000to125_stand,  # SB1 → 125m (Standing start)

            # KPI - Flying専用
            Time000to100,        # Flying: 0m → 100m
            Time100to200,        # Flying: 100m → 200m
            Time000to200,        # Flying: 0m → 200m

            # 速度系KPI
            entry_speed,         # 進入速度 (km/h)
            jump_speed,          # ジャンプ速度 (km/h)

            # 生の時刻データ（各計測地点の通過時刻）
            FP_start,            # Finish/Pursuit Line (ラップ開始)
            SB1,                 # Starting Block 1
            0m_start,            # 0m地点
            60m,                 # 60m地点
            AP1,                 # Apex 1 (62.5m)
            50m,                 # 50m地点
            100m,                # 100m地点
            BP,                  # Banking Point (125m)
            150m,                # 150m地点
            AP2,                 # Apex 2
            200m,                # 200m地点
            FP_2nd,              # 次ラップのFP
            0m_2nd,              # 次ラップの0m
        ]

        ※ 欠損値は距離比例で補間済み
        ※ 全KPI（Rolling/Standing/Flying すべて）を常に計算・含める

    Examples
    --------
    >>> data = get_all_kpis("2025-01-01 14:00:00", "2025-01-01 18:00:00")
    >>> # 選手一覧を取得
    >>> players = list(data.keys())
    >>> print(players)
    ['大田 太郎', '田中 花子', '佐藤 次郎']

    >>> # 特定選手のラップ数を確認
    >>> len(data["大田 太郎"])
    15  # 15ラップ分のデータ

    >>> # 最初のラップのデータを確認
    >>> first_lap = data["大田 太郎"][0]
    >>> print(f"日付: {first_lap[0]}")
    >>> print(f"Time000to625: {first_lap[1]} 秒")
    >>> print(f"FP_start時刻: {first_lap[11]}")

    Notes
    -----
    - settings.json にDB接続情報が正しく設定されている必要があります
    - 期間内にデータが存在しない場合は空の辞書 {} を返します
    - DB接続エラー時は例外が発生します
    """

    # 1. JST → UTC変換
    start_utc = jst_str_to_utc_sql(start_time)
    end_utc = jst_str_to_utc_sql(end_time)

    # 2. 全選手取得用のクエリを生成（user_id指定なし）
    query = f"""
    SELECT
        p.timestamp,
        p.decoder_id,
        p.transponder_id,
        u.first_name,
        u.last_name,
        tu.user_id,
        tu.id AS transponder_user_id
    FROM passing p
    LEFT JOIN transponder_user tu
        ON tu.transponder_id = p.transponder_id
        AND tu.since <= p.timestamp
        AND (tu.until IS NULL OR tu.until > p.timestamp)
    LEFT JOIN `user` u
        ON u.id = tu.user_id
    WHERE p.timestamp >= '{start_utc}'
      AND p.timestamp <  '{end_utc}'
      AND (
            tu.user_id IS NOT NULL              -- ユーザーにひもづく transponder の通過
            OR p.transponder_id IS NULL         -- transponder_id が NULL
            OR p.transponder_id = ''            -- transponder_id が空文字
          )
    ORDER BY p.timestamp
    LIMIT 50000;
    """

    # 3. データ取得＆全KPI計算（既存関数を活用）
    print(f"[KPI API] データ取得開始: {start_time} ～ {end_time}")
    result_df, users = fetch_df_from_db(query, progress=lambda msg: print(f"  → {msg}"))

    # データが空の場合
    if result_df.empty:
        print("[KPI API] 該当期間にデータが存在しません")
        return {}

    print(f"[KPI API] {len(users)} 名の選手、{len(result_df)} ラップを取得")

    # 4. 出力列の定義（順序を明示的に指定）
    output_columns = [
        # 基本情報
        "Date",

        # KPI - Rolling/Standing共通
        "Time000to625",
        "Time625to125",
        "Time125to250",

        # KPI - Rolling専用
        "Time000to125_roll",

        # KPI - Standing専用
        "Time000to125_stand",

        # KPI - Flying専用
        "Time000to100",
        "Time100to200",
        "Time000to200",

        # 速度系
        "entry_speed",
        "jump_speed",

        # 生の時刻データ
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

    # 5. 選手名でグループ化 & List[List]に変換
    output = {}

    for (first_name, last_name), group in result_df.groupby(['first_name', 'last_name'], dropna=False):
        # 選手名の生成（欠損チェック）
        if pd.isna(first_name) or pd.isna(last_name):
            player_name = "Unknown"
        else:
            player_name = f"{last_name} {first_name}"

        # 各ラップをリストに変換
        laps = []
        for _, row in group.iterrows():
            lap_data = []
            for col in output_columns:
                if col in row.index:
                    value = row[col]
                    # pandas特有の型を標準Pythonの型に変換
                    if pd.isna(value):
                        lap_data.append(None)
                    elif isinstance(value, pd.Timestamp):
                        # Timestampは文字列に変換（ミリ秒まで）
                        lap_data.append(value.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
                    elif isinstance(value, (pd.Timedelta, pd._libs.tslibs.timedeltas.Timedelta)):
                        # Timedeltaは秒数(float)に変換
                        lap_data.append(value.total_seconds())
                    else:
                        lap_data.append(value)
                else:
                    # 列が存在しない場合はNone
                    lap_data.append(None)

            laps.append(lap_data)

        output[player_name] = laps

    print(f"[KPI API] 変換完了: {len(output)} 名分のデータ")
    return output


def get_column_info() -> Dict[str, Any]:
    """
    get_all_kpis()が返すデータの列情報を取得

    UI実装時に、どの列が何番目にあるかを確認するためのヘルパー関数です。

    Returns
    -------
    Dict[str, Any]
        列名とインデックスの対応、および説明

    Examples
    --------
    >>> info = get_column_info()
    >>> print(f"Time000to625は {info['columns']['Time000to625']} 番目")
    >>> print(f"FP_startは {info['columns']['FP_start']} 番目")
    """
    columns = [
        "Date",
        "Time000to625",
        "Time625to125",
        "Time125to250",
        "Time000to125_roll",
        "Time000to125_stand",
        "Time000to100",
        "Time100to200",
        "Time000to200",
        "entry_speed",
        "jump_speed",
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

    return {
        "columns": {col: idx for idx, col in enumerate(columns)},
        "total_columns": len(columns),
        "description": {
            "Date": "ラップの日付",
            "Time000to625": "0m → 62.5m(AP1) の所要時間(秒)",
            "Time625to125": "62.5m(AP1) → 125m(BP) の所要時間(秒)",
            "Time125to250": "125m(BP) → 250m(FP_2nd) の所要時間(秒)",
            "Time000to125_roll": "0m → 125m(BP) の所要時間(秒) - Rolling start",
            "Time000to125_stand": "SB1 → 125m(BP) の所要時間(秒) - Standing start",
            "Time000to100": "Flying: 0m → 100m の所要時間(秒)",
            "Time100to200": "Flying: 100m → 200m の所要時間(秒)",
            "Time000to200": "Flying: 0m → 200m の所要時間(秒)",
            "entry_speed": "進入速度 (km/h)",
            "jump_speed": "ジャンプ速度 (km/h)",
            "FP_start": "Finish/Pursuit Line 通過時刻",
            "SB1": "Starting Block 1 通過時刻",
            "0m_start": "0m地点 通過時刻",
            "60m": "60m地点 通過時刻",
            "AP1": "Apex 1 (62.5m) 通過時刻",
            "50m": "50m地点 通過時刻",
            "100m": "100m地点 通過時刻",
            "BP": "Banking Point (125m) 通過時刻",
            "150m": "150m地点 通過時刻",
            "AP2": "Apex 2 通過時刻",
            "200m": "200m地点 通過時刻",
            "FP_2nd": "次ラップのFP 通過時刻",
            "0m_2nd": "次ラップの0m 通過時刻",
        }
    }


def export_to_csv(data, output_path="kpi_data.csv"):
    """KPIデータをCSVに出力"""
    if not data:
        print("データが空です")
        return

    info = get_column_info()
    column_names = list(info['columns'].keys())

    headers = ["選手名", "ラップ番号"] + column_names

    print(f"CSV出力中: {output_path}")

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        total_laps = 0
        for player, laps in data.items():
            for lap_num, lap_data in enumerate(laps, start=1):
                row = [player, lap_num] + lap_data
                writer.writerow(row)
                total_laps += 1

    print(f"✓ 完了: {len(data)} 名、{total_laps} ラップ")

# デバッグ/テスト用
if __name__ == "__main__":
    # 使用例
    print("=" * 60)
    print("KPI API テスト実行")
    print("=" * 60)

    # 列情報の表示
    info = get_column_info()
    print("\n【列情報】")
    print(f"全 {info['total_columns']} 列")
    for col, idx in info['columns'].items():
        desc = info['description'].get(col, "")
        print(f"  [{idx:2d}] {col:20s} - {desc}")

    # データ取得のテスト
    print("\n【データ取得テスト】")
    print("※ settings.json のDB設定が正しいことを確認してください")
    print("※ 実際の日時を指定してテストしてください")
    data = get_all_kpis("2025-09-25 14:00:00", "2025-09-26 18:00:00")
    # CSV出力
    print("\n" + "=" * 60)
    export_to_csv(data, "./kpi_data_demo.csv")


    print("\n" + "=" * 60)
    print("✓ 完了しました")
    print("=" * 60)
    print("\n生成されたファイル:")
    print("  1. /tmp/kpi_data_demo.csv")
    print("  2. /tmp/kpi_columns_info_demo.csv")
    # 例: 実際の日時に置き換えてテスト
    # data = get_all_kpis("2025-09-24 14:00:00", "2025-09-24 18:00:00")
    #
    # if data:
    #     print(f"\n取得した選手: {list(data.keys())}")
    #     for player, laps in data.items():
    #         print(f"  {player}: {len(laps)} ラップ")
    #         if laps:
    #             print(f"    最初のラップ例: {laps[0][:5]}... (最初の5列のみ表示)")
    # else:
    #     print("データが取得できませんでした")