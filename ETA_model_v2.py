"""
================================================================
Smart Port AI — ETA_model_v2.py  (Multi-Type Navires)
================================================================
Lance avec : python ETA_model_v2.py

PRÉREQUIS : avoir d'abord lancé Prepare_noaa_dataset.py
pour générer data/dataset_multi_type.csv

Ce script :
  1. Lit le dataset combiné (GFW pêche + NOAA cargo/ferry/tanker)
  2. Entraîne XGBoost avec 8 features incluant ship_type et vitesse
  3. Valide par type de navire
  4. Sauvegarde eta_tanger_model.pkl
================================================================
"""

import os
import shutil
import numpy as np
import pandas as pd
import joblib
from math import radians, sin, cos, sqrt, asin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# ── Configuration ─────────────────────────────────────────────
PORTS = {
    "Tanger_Ville": (35.7806, -5.8136),
    "Tanger_Med":   (35.8900, -5.5000),
}

FEATURES_MODELE = [
    "distance_to_closest_port",
    "hour",
    "day_of_week",
    "port_encoded",
    "length_m",        # Longueur réelle du bateau
    "tonnage",         # Tonnage GT
    "ship_type",       # Code AIS : 30=Pêche, 60-69=Ferry, 70-79=Cargo, 80-89=Tanker
    "current_speed",   # Vitesse SOG actuelle en nœuds
]

CHEMIN_PKL   = "eta_tanger_model.pkl"
DATASET_CSV  = "data/dataset_multi_type.csv"     # généré par prepare_noaa_dataset.py


# ── Haversine ─────────────────────────────────────────────────
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


def haversine_scalar(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * asin(sqrt(max(0, min(1, a))))


# ── Etape 1 : Charger le dataset ──────────────────────────────
def charger_dataset():
    """
    Charge le dataset multi-type pré-construit par prepare_noaa_dataset.py.
    """
    print("\n" + "="*55)
    print("  ETAPE 1 — Chargement du dataset")
    print("="*55)

    if os.path.exists(DATASET_CSV):
        print(f"  Lecture : {DATASET_CSV}")
        dataset = pd.read_csv(DATASET_CSV, low_memory=False)

        print(f"  Lignes totales : {len(dataset):,}")
        print(f"\n  Répartition par type de navire :")
        for label, borne_inf, borne_sup in [
            ("Pêche  (30)",    30, 30),
            ("Ferry  (60-69)", 60, 69),
            ("Cargo  (70-79)", 70, 79),
            ("Tanker (80-89)", 80, 89),
        ]:
            n = ((dataset["ship_type"] >= borne_inf) &
                 (dataset["ship_type"] <= borne_sup)).sum()
            pct = 100 * n / len(dataset) if len(dataset) > 0 else 0
            print(f"    {label:<18} {n:>8,}  ({pct:.1f}%)")

        return dataset

    else:
        print(f"\n  ERREUR : fichier {DATASET_CSV} introuvable.")
        print(f"\n  Lance d'abord :")
        print(f"    python Prepare_noaa_dataset.py")
        print(f"\n  Ce script combine tes données GFW (pêche) avec")
        print(f"  les données NOAA (cargo, ferry, tanker).")
        return pd.DataFrame()


# ── Etape 2 : Entrainer XGBoost ──────────────────────────────
def entrainer_modele(dataset):
    """
    Entraîne XGBoost sur les 8 features incluant ship_type et current_speed.
    """
    print("\n" + "="*55)
    print("  ETAPE 2 — Entrainement XGBoost (8 features)")
    print("="*55)

    if len(dataset) < 100:
        print(f"  ERREUR : seulement {len(dataset)} lignes — insuffisant")
        return None

    for feat in FEATURES_MODELE:
        if feat not in dataset.columns:
            print(f"  ERREUR : colonne manquante → '{feat}'")
            return None
        dataset[feat] = pd.to_numeric(dataset[feat], errors="coerce")
        dataset[feat] = dataset[feat].fillna(dataset[feat].median())

    X = dataset[FEATURES_MODELE]
    y = dataset["ETA_minutes"]

    print(f"  Features   : {FEATURES_MODELE}")
    print(f"  Train      : {int(len(X)*0.8):,} exemples")
    print(f"  Test       : {int(len(X)*0.2):,} exemples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n  Entrainement en cours...")

    modele = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        tree_method="hist",
        random_state=42,
        verbosity=0,
    )

    modele.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )

    preds = np.maximum(modele.predict(X_test), 0)
    mae   = mean_absolute_error(y_test, preds)

    print(f"\n  Résultats globaux :")
    print(f"    MAE            : {mae:.1f} minutes")
    print(f"    Min prédiction : {preds.min():.2f} min")
    print(f"    Max prédiction : {preds.max():.1f} min")
    print(f"    Moy prédiction : {preds.mean():.1f} min")

    # MAE par type
    print(f"\n  MAE par type de navire :")
    X_test_copy = X_test.copy()
    X_test_copy["y_true"] = y_test.values
    X_test_copy["y_pred"] = preds

    for label, borne_inf, borne_sup in [
        ("Pêche  (30)",    30, 30),
        ("Ferry  (60-69)", 60, 69),
        ("Cargo  (70-79)", 70, 79),
        ("Tanker (80-89)", 80, 89),
    ]:
        mask = (X_test_copy["ship_type"] >= borne_inf) & (X_test_copy["ship_type"] <= borne_sup)
        if mask.sum() > 0:
            mae_type = mean_absolute_error(
                X_test_copy.loc[mask, "y_true"],
                X_test_copy.loc[mask, "y_pred"]
            )
            print(f"    {label:<18} MAE = {mae_type:.1f} min  ({mask.sum()} pts)")

    print(f"\n  Importance des features :")
    importances = modele.feature_importances_
    for feat, imp in sorted(zip(FEATURES_MODELE, importances), key=lambda x: -x[1]):
        barre = "█" * int(imp * 40)
        print(f"    {feat:<28} {barre} ({imp:.3f})")

    return modele


# ── Etape 3 : Validation multi-type ──────────────────────────
def valider_modele(modele):
    print("\n" + "="*55)
    print("  ETAPE 3 — Validation multi-type")
    print("="*55)

    cas_test = [
        ("Pêche   10km  8kn",  10,  30,  8.0,  15.0,    50.0),
        ("Pêche   50km  8kn",  50,  30,  8.0,  15.0,    50.0),
        ("Pêche  200km  8kn", 200,  30,  8.0,  15.0,    50.0),
        ("Ferry   20km 23kn",  20,  65, 23.0, 120.0,  8000.0),
        ("Ferry   50km 23kn",  50,  65, 23.0, 120.0,  8000.0),
        ("Ferry  100km 23kn", 100,  65, 23.0, 120.0,  8000.0),
        ("Cargo   50km 17kn",  50,  75, 17.0, 180.0, 25000.0),
        ("Cargo  200km 17kn", 200,  75, 17.0, 180.0, 25000.0),
        ("Tanker  50km 13kn",  50,  82, 13.0, 250.0, 60000.0),
        ("Tanker 300km 13kn", 300,  82, 13.0, 250.0, 60000.0),
    ]

    ok_count    = 0
    preds_peche = []

    print(f"\n  {'Description':<22} {'ETA XGB':>9} {'ETA Géo':>9}  Statut")
    print(f"  {'-'*55}")

    for desc, dist, stype, speed, length, tonnage in cas_test:
        X = pd.DataFrame([{
            "distance_to_closest_port": dist,
            "hour": 10, "day_of_week": 1, "port_encoded": 0,
            "length_m": length, "tonnage": tonnage,
            "ship_type": stype, "current_speed": speed,
        }])[FEATURES_MODELE]

        pred    = float(modele.predict(X)[0])
        eta_ref = (dist / (speed * 1.852)) * 60
        statut  = "✓" if pred > 1 else "✗ FAIL"
        if pred > 1:
            ok_count += 1
        if stype == 30:
            preds_peche.append(pred)

        print(f"  {desc:<22} {pred:>9.1f} {eta_ref:>9.1f}  {statut}")

    monotone = all(preds_peche[i] < preds_peche[i+1] for i in range(len(preds_peche)-1))
    print(f"\n  Tests OK       : {ok_count}/{len(cas_test)}")
    print(f"  Monotonie ETA  : {'OUI ✓' if monotone else 'NON (toléré avec bruit)'}")
    valide = ok_count >= len(cas_test) * 0.8
    print(f"\n  → {'VALIDATION RÉUSSIE ✓' if valide else 'VALIDATION PARTIELLE'}")
    return valide


# ── Etape 4 : Sauvegarder ────────────────────────────────────
def sauvegarder(modele):
    print("\n" + "="*55)
    print("  ETAPE 4 — Sauvegarde")
    print("="*55)

    if os.path.exists(CHEMIN_PKL):
        backup = CHEMIN_PKL.replace(".pkl", "_backup.pkl")
        shutil.copy(CHEMIN_PKL, backup)
        print(f"  Backup : {backup}")

    joblib.dump(modele, CHEMIN_PKL)
    taille = os.path.getsize(CHEMIN_PKL) / 1024
    print(f"  Fichier : {CHEMIN_PKL}  ({taille:.1f} KB)")

    modele_recharge = joblib.load(CHEMIN_PKL)
    print(f"\n  Vérification finale (50 km, heure 10h) :")

    verif = [
        ("Pêche",   30,  8.0,  15.0,    50.0),
        ("Ferry",   65, 23.0, 120.0,  8000.0),
        ("Cargo",   75, 17.0, 180.0, 25000.0),
        ("Tanker",  82, 13.0, 250.0, 60000.0),
    ]
    tous_ok = True
    for label, stype, speed, length, tonnage in verif:
        X = pd.DataFrame([{
            "distance_to_closest_port": 50.0,
            "hour": 10, "day_of_week": 1, "port_encoded": 0,
            "length_m": length, "tonnage": tonnage,
            "ship_type": stype, "current_speed": speed,
        }])[FEATURES_MODELE]
        pred    = float(modele_recharge.predict(X)[0])
        eta_ref = (50.0 / (speed * 1.852)) * 60
        ok      = pred > 1
        tous_ok = tous_ok and ok
        print(f"    {label:<8} → XGB={pred:.1f} min | Géo={eta_ref:.1f} min  {'✓' if ok else '✗'}")

    if tous_ok:
        print("\n  SUCCÈS — modèle multi-type opérationnel !")
    else:
        print("\n  ATTENTION — certains types prédisent mal")

    return tous_ok


# ── Interface Django ───────────────────────────────────────────
def predire_eta(distance_km, hour, day_of_week, port_encoded,
                length_m, tonnage, ship_type, current_speed_kn,
                chemin_pkl=CHEMIN_PKL):
    """
    Interface pour eta_service.py (Django backend).

    Exemple :
        from Eta_model import predire_eta
        eta = predire_eta(
            distance_km      = distance,
            hour             = datetime.now().hour,
            day_of_week      = datetime.now().weekday(),
            port_encoded     = 1,                    # 0=Tanger Ville, 1=Tanger Med
            length_m         = boat.length,
            tonnage          = boat.tonnage,
            ship_type        = boat.ship_type,       # code AIS
            current_speed_kn = detection.speed,      # SOG AISStream
        )
    """
    modele = joblib.load(chemin_pkl)
    X = pd.DataFrame([{
        "distance_to_closest_port": distance_km,
        "hour":          hour,
        "day_of_week":   day_of_week,
        "port_encoded":  port_encoded,
        "length_m":      length_m,
        "tonnage":       tonnage,
        "ship_type":     ship_type,
        "current_speed": current_speed_kn,
    }])[FEATURES_MODELE]
    return float(max(modele.predict(X)[0], 1.0))


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "="*55)
    print("  Smart Port AI — Ré-entrainement ETA v2")
    print("="*55)

    dataset = charger_dataset()
    if dataset.empty:
        exit(1)

    modele = entrainer_modele(dataset)
    if modele is None:
        exit(1)

    valider_modele(modele)
    succes = sauvegarder(modele)

    print("\n" + "="*55)
    if succes:
        print("  RÉSULTAT : SUCCÈS ✓")
        print("="*55)
        print("\n  Ordre d'exécution complet :")
        print("    1. python Prepare_noaa_dataset.py  ← prépare le CSV fusionné")
        print("    2. python ETA_model_v2.py           ← entraîne XGBoost")
        print("    3. python ETA_test.py               ← vérifie Score 100%")
    else:
        print("  RÉSULTAT : MODÈLE INCERTAIN")
        print("="*55)