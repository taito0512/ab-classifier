import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

# 1. GeoPackageの読み込みと確認
gpkg_path = "data/kasama_ab_labeling.gpkg"
gdf = gpd.read_file(gpkg_path)

print("=== GeoPackage (kasama_ab_labeling.gpkg) ===")
print("行数・列数:", gdf.shape)
print("CRS:", gdf.crs)
print("ジオメトリ種別:", gdf.geom_type.value_counts(dropna=False).to_dict())
print("カラム:", list(gdf.columns))
print(gdf[["id", "label", "done", "geometry"]].head())

# 2. CSVの読み込みと確認
csv_path = "data/kasama_features.csv"
df_csv = pd.read_csv(csv_path)

print("\n=== CSV (data/kasama_features.csv) ===")
print("行数・列数:", df_csv.shape)
print("カラム:", list(df_csv.columns))
print(df_csv[["system:index", "polygon_uu", "point_lat", "point_lng"]].head())
print("先頭行の .geo:", df_csv[".geo"].iloc[0])

# edit_year / issue_year の分布確認
print("\n=== edit_year の分布 ===")
print(df_csv["edit_year"].value_counts(dropna=False).sort_index())

print("\n=== issue_year の分布 ===")
print(df_csv["issue_year"].value_counts(dropna=False).sort_index())

# 3. .geo からポリゴンGeoDataFrameを生成
geometries = [shape(json.loads(geo)) if pd.notna(geo) else None for geo in df_csv[".geo"]]
gdf_poly = gpd.GeoDataFrame(df_csv, geometry=geometries, crs="EPSG:4326")

print("\n=== 空間結合の準備チェック ===")
print("gpkg CRS:", gdf.crs)
print("csvポリゴン仮CRS:", gdf_poly.crs)
print("ポリゴン欠損数:", gdf_poly.geometry.isna().sum())

# 4. メートル単位の平面座標系（EPSG:6677: 平面直角座標系 第IX系・茨城県）に変換して最近傍結合
# gdf の CRS が未設定の場合は WGS84 (EPSG:4326) とみなしてから変換
lab_base = gdf if gdf.crs is not None else gdf.set_crs("EPSG:4326")
lab_p = lab_base.to_crs(6677)
feat_p = gdf_poly.to_crs(6677)

# 各ラベル点に最も近いポリゴンを結合（distance_col で距離[m]も取得）
joined = gpd.sjoin_nearest(
    lab_p,
    feat_p,
    how="left",
    distance_col="dist_m",
)

matched_count = joined["polygon_uu"].notna().sum()
print(f"\nラベル点数: {len(lab_p)} / 最近傍ポリゴン結合数: {matched_count}")
print("結合距離（m）の要約:")
print(joined["dist_m"].describe())


feature_cols = [
    "polygon_uu",
    "NDVI_flood",
    "NDVI_grow",
    "VH_min",
    "VH_winter",
]

# 1. ラベル側と最近傍で紐づいた polygon_uu・距離だけを取り出す
# （等距離で複数ヒットした場合に備え、ラベル点ごとに1件へ絞る場合）
matched_keys = joined[["id", "label", "dist_m", "polygon_uu"]].drop_duplicates(subset=["id"])

# 2. polygon_uu をキーに df_csv とマージ
merged_df = pd.merge(
    matched_keys,
    df_csv[feature_cols],
    on="polygon_uu",
    how="left",
)

print("\n=== マージ結果（先頭5行） ===")
print(merged_df.head())

print("\n=== label の分布（A / B） ===")
print(merged_df["label"].value_counts(dropna=False))

# label が "A" または "B" の行のみに絞り込み（233件になる想定）
df_labeled = merged_df[merged_df["label"].isin(["A", "B"])].copy()

print("\n=== 絞り込み後の件数と label 内訳 ===")
print("総件数:", len(df_labeled))
print(df_labeled["label"].value_counts(dropna=False))

# 特徴量に欠損（NaN）が残っていないか確認
feature_cols = ["NDVI_flood", "NDVI_grow", "VH_min", "VH_winter"]
print("\n=== 特徴量の欠損数 ===")
print(df_labeled[feature_cols].isna().sum())

from pathlib import Path
from sklearn.model_selection import train_test_split

# 1. 結合済みデータの保存
output_path = Path("data/kasama_merged.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df_labeled.to_csv(output_path, index=False)
print(f"\n保存完了: {output_path} ({len(df_labeled)} 行)")

# 2. train / test 分割（層化抽出: stratify=label）
# 233件とやや少なめなので test_size=0.2（約47件）が目安。再現性のため random_state を固定
train_df, test_df = train_test_split(
    df_labeled,
    test_size=0.2,
    stratify=df_labeled["label"],
    random_state=42,
)

print("\n=== 分割結果の確認 ===")
print(f"Train 件数: {len(train_df)}")
print(train_df["label"].value_counts())

print(f"\nTest 件数: {len(test_df)}")
print(test_df["label"].value_counts())