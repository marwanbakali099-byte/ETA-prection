"""
================================================================
Smart Port AI — Prepare_noaa_dataset.py
================================================================
Lance UNE SEULE FOIS pour préparer le dataset combiné :
  - Données GFW  (tes fichiers existants = bateaux de pêche)
  - Données NOAA (CSV téléchargés = cargo, ferry, tanker)

Résultat : data/dataset_multi_type.csv
  → Ce fichier est ensuite lu par ETA_model_v2.py

Structure de ton projet :
  data/
    fleet-daily-csvs-100-v3-2020/
    AIS_2023_01_15/AIS_2023_01_15.csv
    AIS_2023_06_15/AIS_2023_06_15.csv
    AIS_2023_09_15/AIS_2023_09_15.csv
================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, asin

PORTS = {
    "Tanger_Ville": (35.7806, -5.8136),
    "Tanger_Med":   (35.8900, -5.5000),
}

DOSSIER_GFW  = "data/fleet-daily-csvs-100-v3-2020"
DOSSIER_DATA = "data"
SORTIE_CSV   = "data/dataset_multi_type.csv"
DISTANCE_MAX_KM = 600.0

VITESSE_PAR_TYPE = {
    range(0, 30):   8.0,
    range(30, 31):  8.0,
    range(31, 60):  10.0,
    range(60, 70):  23.0,
    range(70, 80):  17.0,
    range(80, 90):  13.0,
    range(90, 100): 10.0,
}

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def traiter_gfw():
    print("\n" + "="*55)
    print("  PARTIE 1 — Données GFW (Pêche)")
    print("="*55)

    fichiers = sorted(glob.glob(os.path.join(DOSSIER_GFW, "*.csv")))[:30]
    if not fichiers:
        print(f"  ATTENTION : aucun fichier GFW dans {DOSSIER_GFW}")
        return pd.DataFrame()

    print(f"  Fichiers trouvés : {len(fichiers)}")
    rows = []

    for i, f in enumerate(fichiers):
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:
            print(f"  Erreur {os.path.basename(f)} : {e}")
            continue

        df = df.rename(columns={
            "cell_ll_lat": "lat", "cell_ll_lon": "lon", "date": "timestamp"
        })
        if "lat" not in df.columns or "lon" not in df.columns:
            continue

        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat", "lon"])

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df["hour"]        = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
        else:
            df["hour"]        = np.random.randint(0, 24, len(df))
            df["day_of_week"] = np.random.randint(0, 7,  len(df))

        df["dist_ville"] = haversine_np(df["lat"].values, df["lon"].values,
                                        PORTS["Tanger_Ville"][0], PORTS["Tanger_Ville"][1])
        df["dist_med"]   = haversine_np(df["lat"].values, df["lon"].values,
                                        PORTS["Tanger_Med"][0],   PORTS["Tanger_Med"][1])

        df["distance_to_closest_port"] = np.minimum(df["dist_ville"], df["dist_med"])
        df["port_encoded"]             = np.where(df["dist_ville"] < df["dist_med"], 0, 1)
        df = df[df["distance_to_closest_port"] < DISTANCE_MAX_KM]
        if df.empty:
            continue

        np.random.seed(42 + i)
        n = len(df)

        df["ship_type"]     = 30
        df["length_m"]      = np.random.choice([8, 12, 15, 18, 22, 28], n,
                                                p=[0.25, 0.20, 0.25, 0.15, 0.10, 0.05])
        df["tonnage"]       = df["length_m"] * np.random.uniform(2.5, 4.0, n)
        df["current_speed"] = 8.0 * (0.85 + 0.30 * np.random.rand(n))

        vitesse_kmh = df["current_speed"].values * 1.852
        coeff_nuit  = np.where((df["hour"].values >= 22) | (df["hour"].values <= 5), 0.80, 1.0)
        vitesse_eff = vitesse_kmh * coeff_nuit
        eta_base    = (df["distance_to_closest_port"].values / vitesse_eff) * 60
        eta_final   = np.maximum(eta_base + np.random.normal(0, eta_base * 0.10), 1.0)
        df["ETA_minutes"] = eta_final
        df = df[(df["ETA_minutes"] > 5) & (df["ETA_minutes"] < 1440)]

        cols = ["distance_to_closest_port", "hour", "day_of_week", "port_encoded",
                "length_m", "tonnage", "ship_type", "current_speed", "ETA_minutes"]
        rows.append(df[cols].dropna())

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    print(f"  → {len(result):,} observations de pêche")
    return result


def traiter_noaa():
    print("\n" + "="*55)
    print("  PARTIE 2 — Données NOAA AIS (Cargo / Ferry / Tanker)")
    print("="*55)

    fichiers = sorted(glob.glob(os.path.join(DOSSIER_DATA, "AIS_*", "*.csv")))

    if not fichiers:
        print(f"  ATTENTION : aucun fichier CSV NOAA trouvé.")
        print(f"  Chemin cherché : {DOSSIER_DATA}/AIS_*/*.csv")
        return pd.DataFrame()

    print(f"  Fichiers trouvés : {len(fichiers)}")
    for f in fichiers:
        print(f"    {f}")

    CODES_CIBLES = list(range(60, 70)) + list(range(70, 80)) + list(range(80, 90))
    rows = []

    for i, f in enumerate(fichiers):
        print(f"  Lecture {os.path.basename(f)}...", end=" ", flush=True)
        try:
            chunks = []
            for chunk in pd.read_csv(f, chunksize=200_000, low_memory=False):
                rename_map = {
                    "LAT": "lat", "LON": "lon",
                    "SOG": "current_speed",
                    "VesselType": "ship_type",
                    "Length": "length_m",
                    "BaseDateTime": "timestamp",
                }
                chunk = chunk.rename(columns=rename_map)

                required = ["lat", "lon", "ship_type", "current_speed"]
                if not all(c in chunk.columns for c in required):
                    continue

                chunk["ship_type"] = pd.to_numeric(chunk["ship_type"], errors="coerce")
                chunk = chunk[chunk["ship_type"].isin(CODES_CIBLES)]
                if chunk.empty:
                    continue

                chunk["lat"]           = pd.to_numeric(chunk["lat"],           errors="coerce")
                chunk["lon"]           = pd.to_numeric(chunk["lon"],            errors="coerce")
                chunk["current_speed"] = pd.to_numeric(chunk["current_speed"], errors="coerce")
                chunk["length_m"]      = pd.to_numeric(chunk.get("length_m",
                                         pd.Series(dtype=float)), errors="coerce")
                chunk = chunk.dropna(subset=["lat", "lon", "ship_type", "current_speed"])
                chunk = chunk[(chunk["current_speed"] >= 0.5) & (chunk["current_speed"] <= 35.0)]
                if chunk.empty:
                    continue
                chunks.append(chunk)

            if not chunks:
                print("aucun enregistrement utile")
                continue

            df = pd.concat(chunks, ignore_index=True)

        except Exception as e:
            print(f"erreur : {e}")
            continue

        print(f"{len(df):,} navires")

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df["hour"]        = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
        else:
            df["hour"]        = np.random.randint(0, 24, len(df))
            df["day_of_week"] = np.random.randint(0, 7,  len(df))

        np.random.seed(100 + i)
        n = len(df)
        distances = np.zeros(n)
        for code_range, dist_min, dist_max in [
            (range(60, 70),  5,  150),
            (range(70, 80), 50,  600),
            (range(80, 90), 50,  600),
        ]:
            masque = df["ship_type"].isin(code_range)
            nb = masque.sum()
            if nb > 0:
                distances[masque] = np.random.uniform(dist_min, dist_max, nb)

        df["distance_to_closest_port"] = distances
        df["port_encoded"] = np.where(df["ship_type"].between(60, 69), 1, 0)

        df["tonnage"] = df["ship_type"].map(
            lambda x: float(np.random.uniform(
                1000 if 60<=x<=69 else 3000 if 70<=x<=79 else 5000,
                40000 if 60<=x<=69 else 80000 if 70<=x<=79 else 150000
            ))
        )

        for code_range, val_def in [(range(60,70), 120.0), (range(70,80), 180.0), (range(80,90), 250.0)]:
            for code in code_range:
                masque = (df["ship_type"] == code) & df["length_m"].isna()
                if masque.sum() > 0:
                    df.loc[masque, "length_m"] = val_def + np.random.normal(0, val_def*0.15, masque.sum())
        df["length_m"] = df["length_m"].fillna(100.0)

        vitesse_kmh = df["current_speed"].values * 1.852
        coeff_nuit  = np.where((df["hour"].values >= 22) | (df["hour"].values <= 5), 0.85, 1.0)
        vitesse_eff = np.maximum(vitesse_kmh * coeff_nuit, 1.0)
        eta_base    = (df["distance_to_closest_port"].values / vitesse_eff) * 60
        bruit       = np.random.normal(0, eta_base * 0.08)
        df["ETA_minutes"] = np.maximum(eta_base + bruit, 1.0)
        df = df[(df["ETA_minutes"] > 1) & (df["ETA_minutes"] < 1440)]

        cols = ["distance_to_closest_port", "hour", "day_of_week", "port_encoded",
                "length_m", "tonnage", "ship_type", "current_speed", "ETA_minutes"]
        df_clean = df[cols].dropna()

        for label, b0, b1 in [("Ferry(60-69)",60,69),("Cargo(70-79)",70,79),("Tanker(80-89)",80,89)]:
            n_t = ((df_clean["ship_type"] >= b0) & (df_clean["ship_type"] <= b1)).sum()
            if n_t > 0:
                print(f"    {label} : {n_t:,} obs")

        rows.append(df_clean)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    print(f"\n  → {len(result):,} observations NOAA au total")
    return result


def fusionner_et_sauvegarder(df_gfw, df_noaa):
    print("\n" + "="*55)
    print("  PARTIE 3 — Fusion et sauvegarde")
    print("="*55)

    frames = [df for df in [df_gfw, df_noaa] if not df.empty]
    if not frames:
        print("ERREUR : aucune donnée disponible")
        return

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nDataset combiné : {len(dataset):,} observations")
    for label, b0, b1 in [("Pêche(30)",30,30),("Ferry(60-69)",60,69),
                           ("Cargo(70-79)",70,79),("Tanker(80-89)",80,89)]:
        n = ((dataset["ship_type"] >= b0) & (dataset["ship_type"] <= b1)).sum()
        if n > 0:
            pct = 100 * n / len(dataset)
            print(f"  {label:<18} {n:>8,}  ({pct:.1f}%)")

    os.makedirs(os.path.dirname(SORTIE_CSV), exist_ok=True)
    dataset.to_csv(SORTIE_CSV, index=False)
    taille = os.path.getsize(SORTIE_CSV) / (1024 * 1024)
    print(f"\nFichier sauvegardé : {SORTIE_CSV}  ({taille:.1f} MB)")
    print(f"\n→ Lance maintenant : python ETA_model_v2.py")


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Smart Port AI — Préparation Dataset Multi-Type")
    print("="*55)

    gfw_ok = os.path.exists(DOSSIER_GFW)
    noaa_ok = len(glob.glob(os.path.join(DOSSIER_DATA, "AIS_*", "*.csv"))) > 0

    print(f"\n  GFW  : {'✓ OK' if gfw_ok  else '✗ absent'}")
    print(f"  NOAA : {'✓ OK' if noaa_ok else '✗ aucun fichier AIS_* trouvé'}")

    if not gfw_ok and not noaa_ok:
        print("\n  Aucune donnée. Abandonne.")
        exit(1)

    df_gfw  = traiter_gfw()  if gfw_ok  else pd.DataFrame()
    df_noaa = traiter_noaa() if noaa_ok else pd.DataFrame()

    fusionner_et_sauvegarder(df_gfw, df_noaa)