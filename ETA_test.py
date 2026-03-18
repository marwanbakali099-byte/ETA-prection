"""
================================================================
Smart Port AI — ETA_test.py  (v2 — Multi-Type Navires)
================================================================
Lance avec : python ETA_test.py
Compatible avec ETA_model_v2.py (8 features)
================================================================
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import joblib
import time
from math import radians, sin, cos, sqrt, asin

# ── Couleurs terminal ─────────────────────────────────────────
VERT  = "\033[92m"
ROUGE = "\033[91m"
JAUNE = "\033[93m"
BLEU  = "\033[94m"
GRAS  = "\033[1m"
RESET = "\033[0m"
OK    = f"{VERT}[  OK  ]{RESET}"
FAIL  = f"{ROUGE}[ FAIL ]{RESET}"
WARN  = f"{JAUNE}[ WARN ]{RESET}"
INFO  = f"{BLEU}[ INFO ]{RESET}"

score = {"ok": 0, "fail": 0, "warn": 0}

# ── 8 features du nouveau modèle ─────────────────────────────
FEATURES_MODELE = [
    "distance_to_closest_port",
    "hour",
    "day_of_week",
    "port_encoded",
    "length_m",
    "tonnage",
    "ship_type",
    "current_speed",
]

PORTS = {
    "Tanger_Ville": (35.7806, -5.8136),
    "Tanger_Med":   (35.8900, -5.5000),
}

# Profils navires pour les tests
PROFILS = {
    "peche":  {"ship_type": 30,  "speed": 8.0,  "length": 15.0,  "tonnage": 50.0},
    "ferry":  {"ship_type": 65,  "speed": 23.0, "length": 120.0, "tonnage": 8000.0},
    "cargo":  {"ship_type": 75,  "speed": 17.0, "length": 180.0, "tonnage": 25000.0},
    "tanker": {"ship_type": 82,  "speed": 13.0, "length": 250.0, "tonnage": 60000.0},
}


def titre(t):
    print(f"\n{GRAS}{'=' * 60}{RESET}")
    print(f"{GRAS}  {t}{RESET}")
    print(f"{GRAS}{'=' * 60}{RESET}")


def sous_titre(t):
    print(f"\n{BLEU}--- {t} ---{RESET}")


def check(cond, msg_ok, msg_fail, warn=False):
    if cond:
        print(f"  {OK}  {msg_ok}")
        score["ok"] += 1
    elif warn:
        print(f"  {WARN}  {msg_fail}")
        score["warn"] += 1
    else:
        print(f"  {FAIL}  {msg_fail}")
        score["fail"] += 1


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * asin(sqrt(max(0, min(1, a))))


def port_proche(lat, lon):
    d = {n: haversine(lat, lon, c[0], c[1]) for n, c in PORTS.items()}
    p = min(d, key=d.get)
    return p, d[p]


def eta_geo(lat, lon, vitesse):
    if vitesse < 0.5:
        return -1, None, None
    p, dist = port_proche(lat, lon)
    eta = (dist / (vitesse * 1.852)) * 60
    return round(eta, 1), round(dist, 2), p


def predire_xgb(modele, dist, hour=10, day=1, port_enc=0,
                length=15.0, tonnage=50.0,
                ship_type=30, current_speed=8.0):
    """
    Prédiction XGBoost avec les 8 features du nouveau modèle.
    """
    X = pd.DataFrame([{
        "distance_to_closest_port": dist,
        "hour":          hour,
        "day_of_week":   day,
        "port_encoded":  port_enc,
        "length_m":      length,
        "tonnage":       tonnage,
        "ship_type":     ship_type,
        "current_speed": current_speed,
    }])[FEATURES_MODELE]
    return float(modele.predict(X)[0])


# =============================================================
# TEST 1 — Fichiers du projet
# =============================================================
titre("TEST 1 — Verification des fichiers")

check(os.path.exists("ETA_model_v2.py"),
      "ETA_model_v2.py trouve", "ETA_model_v2.py INTROUVABLE")
check(os.path.exists("Prepare_noaa_dataset.py"),
      "Prepare_noaa_dataset.py trouve", "Prepare_noaa_dataset.py INTROUVABLE", warn=True)
check(os.path.exists("eta_tanger_model.pkl"),
      "eta_tanger_model.pkl trouve", "pkl introuvable")
check(os.path.exists("data/dataset_multi_type.csv"),
      "dataset_multi_type.csv trouve",
      "dataset_multi_type.csv absent — lance Prepare_noaa_dataset.py", warn=True)
check(os.path.exists("data/fishing-vessels-v3.csv"),
      "fishing-vessels-v3.csv trouve", "vessels csv introuvable", warn=True)

fichiers_gfw = glob.glob("data/fleet-daily-csvs-100-v3-2020/*.csv")
check(len(fichiers_gfw) > 0,
      f"{len(fichiers_gfw)} fichiers CSV GFW trouves", "Aucun fichier CSV GFW")

fichiers_noaa = glob.glob("data/AIS_*/*.csv")
check(len(fichiers_noaa) > 0,
      f"{len(fichiers_noaa)} fichiers CSV NOAA trouves",
      "Aucun fichier NOAA (data/AIS_*/)", warn=True)


# =============================================================
# TEST 2 — Chargement du modele PKL
# =============================================================
titre("TEST 2 — Chargement du modele PKL")

modele = None

if os.path.exists("eta_tanger_model.pkl"):
    sous_titre("Lecture du fichier")
    try:
        contenu = joblib.load("eta_tanger_model.pkl")
        modele  = contenu[0] if isinstance(contenu, tuple) else contenu
        print(f"  {INFO}  Format : {'(modele, features)' if isinstance(contenu, tuple) else 'modele seul'}")

        check(modele is not None, "Modele charge avec succes", "Erreur chargement")
        print(f"  {INFO}  n_estimators : {getattr(modele, 'n_estimators', '?')}")
        print(f"  {INFO}  max_depth    : {getattr(modele, 'max_depth', '?')}")

        if hasattr(modele, 'feature_names_in_'):
            features_reelles = list(modele.feature_names_in_)
            print(f"  {INFO}  Features du modele : {features_reelles}")

            check(
                len(features_reelles) == 8,
                "8 features (v2 multi-type) — correct",
                f"{len(features_reelles)} features — attendu 8 (relance ETA_model_v2.py)"
            )
            check(
                "ship_type" in features_reelles,
                "ship_type present — modele v2 confirme",
                "ship_type absent — ancien modele detecte"
            )
            check(
                "current_speed" in features_reelles,
                "current_speed present — correct",
                "current_speed absent — ancien modele"
            )
            # Vérifier absence des anciennes features
            old_feats = [f for f in ["length_m_gfw", "tonnage_gt_gfw", "speed", "distance_delta"]
                         if f in features_reelles]
            check(
                len(old_feats) == 0,
                "Aucune ancienne feature (length_m_gfw etc.) — propre",
                f"Anciennes features trouvées : {old_feats} — relance ETA_model_v2.py"
            )

    except Exception as e:
        print(f"  {FAIL}  Erreur : {e}")
        score["fail"] += 1
else:
    print(f"  {WARN}  Modele PKL non trouve")
    score["warn"] += 1


# =============================================================
# TEST 3 — Predictions par type de navire
# =============================================================
titre("TEST 3 — Predictions par type de navire (50 km)")

if modele is not None:
    print(f"\n  {'Type':<10} {'Code AIS':<10} {'Vitesse':<10} {'ETA XGB':<12} {'ETA Geo':<12} {'Statut'}")
    print(f"  {'-'*64}")

    for type_nom, p in PROFILS.items():
        try:
            eta_x   = predire_xgb(modele, 50.0,
                                   ship_type=p["ship_type"],
                                   current_speed=p["speed"],
                                   length=p["length"],
                                   tonnage=p["tonnage"])
            eta_ref = (50.0 / (p["speed"] * 1.852)) * 60
            ok      = eta_x > 1
            statut  = f"{VERT}✓{RESET}" if ok else f"{ROUGE}✗{RESET}"
            print(f"  {type_nom:<10} {p['ship_type']:<10} {p['speed']:<10.1f} "
                  f"{eta_x:<12.1f} {eta_ref:<12.1f} {statut}")
            check(ok,
                  f"{type_nom} ETA={eta_x:.1f} min — OK",
                  f"{type_nom} ETA={eta_x:.2f} min — FAIL")
        except Exception as e:
            print(f"  {type_nom:<10} ERREUR : {e}")
            score["fail"] += 1
else:
    print(f"  {WARN}  Modele non disponible")
    score["warn"] += 1


# =============================================================
# TEST 4 — Predictions XGBoost sur plusieurs distances
# =============================================================
titre("TEST 4 — Predictions XGBoost (distances variables, pêche)")

if modele is not None:
    sous_titre("Bateau de pêche — distances 10 à 200 km")

    distances_test = [10, 30, 50, 100, 150, 200]
    preds_xgb      = []

    print(f"\n  {'Distance':<12} {'ETA XGB':<14} {'ETA Geo':<14} {'Diff'}")
    print(f"  {'-'*55}")

    for dist in distances_test:
        try:
            eta_x = predire_xgb(modele, dist,
                                 ship_type=30, current_speed=8.0,
                                 length=15.0, tonnage=50.0)
            preds_xgb.append(eta_x)
            eta_ref = (dist / (8.0 * 1.852)) * 60
            diff    = abs(eta_x - eta_ref)
            print(f"  {dist:<12} {eta_x:<14.1f} {eta_ref:<14.1f} {diff:.0f}")
        except Exception as e:
            print(f"  {dist:<12} ERREUR : {e}")
            preds_xgb.append(0)

    if preds_xgb:
        arr = np.array(preds_xgb)
        check(all(p > 0 for p in preds_xgb),
              "Toutes predictions positives",
              f"Predictions nulles : {preds_xgb}")
        check(arr.std() > 1,
              f"Predictions variables (std={arr.std():.1f})",
              f"Predictions constantes (std={arr.std():.1f})", warn=True)
        corr = np.corrcoef(distances_test, preds_xgb)[0, 1]
        check(corr > 0.8,
              f"Correlation distance/ETA = {corr:.2f} — excellent",
              f"Correlation = {corr:.2f} — faible", warn=corr > 0.5)
else:
    print(f"  {WARN}  Modele non disponible")
    score["warn"] += 1


# =============================================================
# TEST 5 — Formule géométrique Haversine
# =============================================================
titre("TEST 5 — Formule geometrique Haversine")

sous_titre("Positions autour de Tanger")

positions = [
    (35.60, -5.88,  8.0, "Entree port Tanger"),
    (35.50, -5.95, 10.0, "Baie de Tanger"),
    (35.30, -6.10, 12.0, "Au large Tanger"),
    (36.00, -5.60,  8.0, "Detroit de Gibraltar"),
    (35.78, -5.81,  1.0, "Dans le port"),
    (35.60, -5.88,  0.0, "Bateau a l'arret"),
]

print(f"\n  {'Nom':<25} {'Dist(km)':<12} {'ETA(min)':<12} {'Port'}")
print(f"  {'-'*65}")

for lat, lon, vit, nom in positions:
    eta, dist, port = eta_geo(lat, lon, vit)
    if eta == -1:
        print(f"  {nom:<25} {'--':<12} {'ARRET':<12} --")
    else:
        print(f"  {nom:<25} {dist:<12.1f} {eta:<12.1f} {port}")
    check(eta != 0.0 or vit < 0.5, f"{nom} correct", f"{nom} ETA=0 probleme")


# =============================================================
# TEST 6 — Flotte multi-type (pêche + cargo + ferry)
# =============================================================
titre("TEST 6 — Simulation flotte multi-type")

flotte = [
    {"mmsi": "212345678", "nom": "Al Amal",     "lat": 35.40, "lon": -5.95,
     "v": 9.5,  "ship_type": 30,  "length": 15.0,  "tonnage": 50.0},
    {"mmsi": "338234756", "nom": "Badr",         "lat": 35.55, "lon": -5.88,
     "v": 7.0,  "ship_type": 30,  "length": 12.0,  "tonnage": 40.0},
    {"mmsi": "224567890", "nom": "Atlas Ferry",  "lat": 35.90, "lon": -5.55,
     "v": 22.0, "ship_type": 65,  "length": 130.0, "tonnage": 9000.0},
    {"mmsi": "227654321", "nom": "MedCargo",     "lat": 35.20, "lon": -6.10,
     "v": 16.0, "ship_type": 72,  "length": 185.0, "tonnage": 30000.0},
    {"mmsi": "215432109", "nom": "Gulf Tanker",  "lat": 35.50, "lon": -5.70,
     "v": 13.0, "ship_type": 84,  "length": 240.0, "tonnage": 55000.0},
]

print(f"\n  {'MMSI':<12} {'Nom':<14} {'Type':<8} {'Dist':<8} {'Geo':<8} {'XGB':<8} {'Méthode'}")
print(f"  {'-'*72}")

for b in flotte:
    eta_g, dist, port = eta_geo(b["lat"], b["lon"], b["v"])
    eta_x_str = "--"
    methode   = "geometrique"

    if modele is not None and eta_g != -1 and dist:
        try:
            p, _ = port_proche(b["lat"], b["lon"])
            pe   = list(PORTS.keys()).index(p)
            val  = predire_xgb(modele, dist,
                               port_enc=pe,
                               ship_type=b["ship_type"],
                               current_speed=b["v"],
                               length=b["length"],
                               tonnage=b["tonnage"])
            if 0 < val < 1440:
                eta_x_str = f"{val:.0f}"
                methode   = "xgboost"
            else:
                eta_x_str = "ERR"
        except:
            eta_x_str = "ERR"

    type_label = {30: "Pêche", 65: "Ferry", 72: "Cargo", 84: "Tanker"}.get(b["ship_type"], "?")
    eta_str    = f"{eta_g}" if eta_g != -1 else "ARRET"
    print(f"  {b['mmsi']:<12} {b['nom']:<14} {type_label:<8} {dist or '--':<8} "
          f"{eta_str:<8} {eta_x_str:<8} {methode}")
    check(eta_g != 0.0 or b["v"] < 0.5,
          f"{b['nom']} OK ({type_label})", f"{b['nom']} ETA=0 probleme")


# =============================================================
# TEST 7 — Format JSON Django DRF
# =============================================================
titre("TEST 7 — Format JSON API Django DRF")


def api_eta(lat, lon, vitesse, mmsi, nom,
            ship_type=30, length=15.0, tonnage=50.0):
    eta_g, dist, port = eta_geo(lat, lon, vitesse)
    methode   = "geometrique"
    eta_final = eta_g

    if modele is not None and eta_g != -1 and dist:
        try:
            p, _ = port_proche(lat, lon)
            pe   = list(PORTS.keys()).index(p)
            val  = predire_xgb(modele, dist,
                               port_enc=pe,
                               ship_type=ship_type,
                               current_speed=vitesse,
                               length=length,
                               tonnage=tonnage)
            if 0 < val < 1440:
                eta_final = round(val, 1)
                methode   = "xgboost_ml"
        except:
            pass

    return {
        "mmsi":        mmsi,
        "vessel_name": nom,
        "ship_type":   ship_type,
        "latitude":    lat,
        "longitude":   lon,
        "speed_knots": vitesse,
        "eta_minutes": eta_final,
        "eta_status":  "bateau_arret" if eta_final == -1 else "en_route",
        "distance_km": dist,
        "port_cible":  port,
        "methode":     methode,
        "message": (
            "Bateau a l'arret" if eta_final == -1
            else f"Arrivee prevue dans {eta_final:.0f} minutes"
        )
    }


bateaux_api = [
    {"lat": 35.50, "lon": -5.90, "v":  9.0, "mmsi": "212345678", "nom": "Al Amal",
     "ship_type": 30,  "length": 15.0,  "tonnage": 50.0},
    {"lat": 35.90, "lon": -5.55, "v": 22.0, "mmsi": "338234756", "nom": "Atlas Ferry",
     "ship_type": 65,  "length": 130.0, "tonnage": 9000.0},
    {"lat": 35.78, "lon": -5.81, "v":  0.3, "mmsi": "224567890", "nom": "Nour",
     "ship_type": 30,  "length": 12.0,  "tonnage": 40.0},
]

for b in bateaux_api:
    rep = api_eta(b["lat"], b["lon"], b["v"], b["mmsi"], b["nom"],
                  ship_type=b["ship_type"], length=b["length"], tonnage=b["tonnage"])
    print(f"\n  {BLEU}{rep['vessel_name']} ({rep['mmsi']}) — ship_type={rep['ship_type']}{RESET}")
    for k, v in rep.items():
        print(f"    {k:<16} : {v}")
    check(
        rep["eta_minutes"] != 0.0 or rep["eta_status"] == "bateau_arret",
        f"{rep['vessel_name']} ETA valide via {rep['methode']}",
        f"{rep['vessel_name']} ETA = 0.0 PROBLEME"
    )


# =============================================================
# TEST 8 — Performance
# =============================================================
titre("TEST 8 — Performance")

t0 = time.time()
for _ in range(1000):
    eta_geo(np.random.uniform(34.5, 36.5),
            np.random.uniform(-7.0, -4.5),
            np.random.uniform(1, 20))
duree = time.time() - t0
check(duree < 2.0,
      f"1000 predictions geometriques en {duree*1000:.0f} ms",
      f"Trop lent : {duree:.2f}s", warn=duree < 3.0)

if modele is not None:
    t1       = time.time()
    ok_count = 0
    types    = [30, 65, 75, 82]
    for j in range(100):
        try:
            st = types[j % 4]
            sp = {30: 8.0, 65: 23.0, 75: 17.0, 82: 13.0}[st]
            predire_xgb(modele,
                        np.random.uniform(10, 300),
                        hour=np.random.randint(0, 23),
                        day=np.random.randint(0, 6),
                        ship_type=st,
                        current_speed=sp * np.random.uniform(0.8, 1.2))
            ok_count += 1
        except:
            pass
    duree2 = time.time() - t1
    check(ok_count == 100,
          f"100/100 predictions XGBoost (4 types) en {duree2*1000:.0f} ms",
          f"Echecs XGBoost : {100 - ok_count}/100")


# =============================================================
# RESUME FINAL
# =============================================================
titre("RESUME FINAL")

total = score["ok"] + score["fail"] + score["warn"]
taux  = (score["ok"] / total * 100) if total > 0 else 0

print(f"\n  {VERT}Tests reussis  : {score['ok']}{RESET}")
print(f"  {JAUNE}Warnings       : {score['warn']}{RESET}")
print(f"  {ROUGE}Tests echoues  : {score['fail']}{RESET}")
print(f"\n  Score          : {taux:.0f}%  ({score['ok']}/{total})")

if modele is not None:
    print(f"\n{GRAS}  Vérification finale par type (50 km) :{RESET}")
    try:
        for type_nom, p in PROFILS.items():
            pred = predire_xgb(modele, 50.0,
                               ship_type=p["ship_type"],
                               current_speed=p["speed"],
                               length=p["length"],
                               tonnage=p["tonnage"])
            eta_ref = (50.0 / (p["speed"] * 1.852)) * 60
            couleur = VERT if pred > 1 else ROUGE
            print(f"  {couleur}  {type_nom:<8} (code={p['ship_type']}) → "
                  f"XGB={pred:.1f} min | Géo={eta_ref:.1f} min{RESET}")
    except Exception as e:
        print(f"  {ROUGE}  Erreur : {e}{RESET}")

if score["fail"] == 0:
    print(f"\n  {VERT}{GRAS}  Tous les tests passes ! Score 100%{RESET}")
    print(f"\n  Intégration Django (eta_service.py) :")
    print(f"    from ETA_model_v2 import predire_eta")
    print(f"    eta = predire_eta(")
    print(f"        distance_km      = dist,")
    print(f"        hour             = datetime.now().hour,")
    print(f"        day_of_week      = datetime.now().weekday(),")
    print(f"        port_encoded     = 1,")
    print(f"        length_m         = boat.length,")
    print(f"        tonnage          = boat.tonnage,")
    print(f"        ship_type        = boat.ship_type,")
    print(f"        current_speed_kn = detection.speed,")
    print(f"    )")
else:
    print(f"\n  {JAUNE}  {score['fail']} test(s) échoué(s) — voir détails ci-dessus{RESET}")

print()