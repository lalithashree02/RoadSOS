from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import os

app = Flask(__name__)
CORS(app)

# ============================================================
# ENDPOINT API ROUTES (FRONTEND ROUTING)
# ============================================================

@app.route("/")
def landing_page():
    """Serves the main marketing landing page first."""
    return render_template("landing.html")


@app.route("/app")
def dashboard_app():
    """Serves the interactive live tracking map interface dashboard."""
    return render_template("index.html")


@app.route("/quick-sos")
def rapid_sos_utility():
    """Serves the compact offline-first utility interface panel."""
    return render_template("utility-sos.html")

# ============================================================
# FILE PATHS  
# ============================================================
HOSPITALS_FILE = os.path.join("data", "hospitals.csv")
PUNCTURE_FILE  = os.path.join("data", "puncture_1.csv")
TOWSHOPS_FILE  = os.path.join("data", "Towshops_India.csv")
POLICE_FILE    = os.path.join("data", "police.csv")


# ============================================================
# CATEGORY NORMALISER
# ============================================================
def normalize_category(row):
    """
    Safely resolves and maps raw data attributes to one of our 
    6 core target categories: hospital, fuel, mechanic, towing, police, air.
    """
    raw_cat  = str(row.get("category",         "") or "").lower().strip()
    raw_type = str(row.get("type",             "") or "").lower().strip()
    raw_svc  = str(row.get("services_offered", "") or "").lower().strip()
    raw_name = str(row.get("name",             "") or "").lower().strip()
    raw_addr = str(row.get("address",          "") or "").lower().strip()

    # Category Alias Lookup Mapping
    CAT_ALIASES = {
        "towshop":    "towing",
        "tow_shop":   "towing",
        "tow shop":   "towing",
        "towing":     "towing",
        "puncture":   "air",
        "tyre":       "air",
        "tire":       "air",
        "air":        "air",
        "petrol":     "fuel",
        "gas":        "fuel",
        "fuel":       "fuel",
        "medical":    "hospital",
        "clinic":     "hospital",
        "health":     "hospital",
        "trauma":     "hospital",
        "hospital":   "hospital",
        "police":     "police",
        "mechanic":   "mechanic"
    }

    # 1. Check explicitly provided category column or standard aliases
    if raw_cat in CAT_ALIASES:
        return CAT_ALIASES[raw_cat]
        
    # 2. Check the secondary feature "type" column (often found in hospital sheets)
    if raw_type in CAT_ALIASES:
        return CAT_ALIASES[raw_type]

    # 3. Fallback to keyword scanning across combined properties
    combined = f"{raw_type} {raw_svc} {raw_name} {raw_addr}"

    if any(k in combined for k in ["police", "cop", "thana", "rpf", "chowki"]):
        return "police"
    if any(k in combined for k in ["tow", "towing", "towshop", "rescue", "breakdown", "crane"]):
        return "towing"
    if any(k in combined for k in ["tyre", "tire", "puncture", "vulcaniz", "tubeless", "wheel", "air"]):
        return "air"
    if any(k in combined for k in ["hospital", "clinic", "health", "trauma", "medical", "nursing", "dispensary"]):
        return "hospital"
    if any(k in combined for k in ["petrol", "diesel", "fuel", "gas station", "bunk", "pump"]):
        return "fuel"
    if any(k in combined for k in ["mechanic", "garage", "workshop", "service center", "repair"]):
        return "mechanic"

    return "hospital"


# ============================================================
# HAVERSINE DISTANCE  (km, rounded to 2 dp)
# ============================================================
def haversine(lat1, lon1, lat2, lon2):
    try:
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return round(R * 2 * atan2(sqrt(a), sqrt(1 - a)), 2)
    except (ValueError, TypeError):
        return 99999.0  # Safe fallback high distance value for broken coordinate sets


# ============================================================
# CSV LOADER
# ============================================================
def load_csv(filepath, default_category):
    """
    Loads one CSV file, handles missing headers or separator issues automatically,
    normalizes varying headers, and cleans coordinates.
    """
    if not os.path.exists(filepath):
        print(f"  ⚠️  File not found, skipping configuration: {filepath}")
        return pd.DataFrame()

    try:
        # --- SMART DELIMITER DETECTION ---
        sep = ','
        with open(filepath, 'r', encoding='latin1') as f:
            first_line = f.readline()
            
        # If there are actual commas separating items, stick to comma
        if ',' in first_line and first_line.count(',') >= first_line.count('\t'):
            sep = ','
        elif '\t' in first_line:
            sep = '\t'
        elif ';' in first_line:
            sep = ';'

        # --- CHECK FOR MISSING HEADERS (hospitals.csv fallback) ---
        first_val = first_line.split(sep)[0].strip().lower()
        
        if filepath.endswith("hospitals.csv") and (first_val.isdigit() or "599" in first_val or "id" not in first_val):
            column_names = [
                "id", "element", "lat", "lon", "name", "address_id", 
                "address", "city", "state", "country", "type", "timings", "category"
            ]
            df = pd.read_csv(filepath, sep=sep, names=column_names, header=None, encoding="latin1", dtype=str)
        else:
            df = pd.read_csv(filepath, sep=sep, encoding="latin1", dtype=str)
        
        # Clean up column names (strip spaces, lowercase, remove quotes/tabs)
        df.columns = [c.strip().lower().replace('"', '').replace("'", "") for c in df.columns]

        # Drop any accidental trailing blank columns like 'unnamed: 1'
        df = df.loc[:, ~df.columns.str.contains('^unnamed')]

        # ── Rename dynamic variation rules seen in your raw datasets ──
        rename_map = {}
        
        if "latitude" in df.columns:   rename_map["latitude"] = "lat"
        if "longitude" in df.columns:  rename_map["longitude"] = "lon"
        if "lng" in df.columns:        rename_map["lng"] = "lon"
        
        if "name" not in df.columns:
            name_cols = [c for c in df.columns if "hospital" in c and "name" in c]
            if name_cols:                  rename_map[name_cols[0]] = "name"
            elif "station_name" in df.columns: rename_map["station_name"] = "name"
            elif "shop_name" in df.columns:    rename_map["shop_name"] = "name"
                
        if "address" not in df.columns and "full_address" in df.columns: rename_map["full_address"] = "address"
        if "phone"   not in df.columns and "phone_number" in df.columns: rename_map["phone_number"] = "phone"
        if "phone"   not in df.columns and "helpline_number" in df.columns: rename_map["helpline_number"] = "phone"
        if "timings" not in df.columns and "opening_hours" in df.columns: rename_map["opening_hours"] = "timings"
        if "city"    not in df.columns and "district" in df.columns:      rename_map["district"] = "city"
        
        df.rename(columns=rename_map, inplace=True)

        # Confirm critical coordinate mappings are present
        if "lat" not in df.columns or "lon" not in df.columns:
            print(f"  ❌ Missing critical lat/lon column names in {filepath}. Available headers: {list(df.columns)}")
            return pd.DataFrame()

        # Ensure all expected application framework columns exist
        core_columns = ["id", "name", "lat", "lon", "address", "city", "state",
                        "phone", "category", "timings", "type", "services_offered"]
        for col in core_columns:
            if col not in df.columns:
                df[col] = ""

        # Coordinate numerical casting and validation data sanitization
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat", "lon"]).copy()

        # Handle category defaults safely
        df["category"] = df["category"].fillna("")
        df.loc[df["category"].astype(str).str.strip() == "", "category"] = default_category
        df["category"] = df.apply(normalize_category, axis=1)

        print(f"  ✅ {len(df):>6} records  ←  {filepath}")
        return df

    except Exception as e:
        print(f"  ❌ Error running parser over {filepath}: {e}")
        return pd.DataFrame()







# ============================================================
# LOAD ALL DATA
# ============================================================
def load_all_data():
    print("\n📂 Initializing pipeline systems loading infrastructure...")

    hospitals = load_csv(HOSPITALS_FILE, default_category="hospital")
    puncture  = load_csv(PUNCTURE_FILE,  default_category="air")
    towshops  = load_csv(TOWSHOPS_FILE,  default_category="towing")
    police    = load_csv(POLICE_FILE,    default_category="police")

    frames = [df for df in [hospitals, puncture, towshops, police] if not df.empty]

    if not frames:
        print("  ❌ Critical Initialization Error: No active tracking files located.")
        return pd.DataFrame(columns=[
            "id", "name", "lat", "lon", "address", "city", "state",
            "phone", "category", "timings", "type", "services_offered"
        ])

    combined = pd.concat(frames, ignore_index=True)
    combined["id"] = combined.index.astype(str)   # Auto-generates unique system-level tracking keys

    # Print out summary statistics during initialization 
    cat_counts = combined["category"].value_counts().to_dict()
    print(f"\n  📊 Compiled Server Metrics Report:")
    print(f"     Total Active Records Found: {len(combined)}")
    for cat, count in sorted(cat_counts.items()):
        print(f"     • {cat:<12}: {count}")
    print()

    return combined


# ============================================================
# ROW → JSON CONVERTER
# ============================================================
def row_to_dict(row, distance_km):
    address = str(row.get("address", "") or "").strip()
    city    = str(row.get("city",    "") or "").strip()

    if city and city.lower() not in address.lower():
        address = f"{address}, {city}".strip(", ")

    return {
        "id":               str(row.get("id", "")),
        "name":             str(row.get("name", "Unknown System Node") or "Unknown System Node").strip(),
        "category":         str(row.get("category", "hospital")),
        "lat":              float(row["lat"]),
        "lon":              float(row["lon"]),
        "address":          address,
        "city":             city,
        "state":            str(row.get("state",   "") or "").strip(),
        "phone":            str(row.get("phone",   "") or "").strip(),
        "timings":          str(row.get("timings", "") or "").strip(),
        "services_offered": str(row.get("services_offered", "") or "").strip(),
        "distance":         distance_km,
    }


# Load full global system database into engine memory
service_df = load_all_data()


# ============================================================
# ENDPOINT API ROUTES
# ============================================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/nearest-hospitals")    # Kept intact for backward compatibility integrations
@app.route("/api/nearest-services")
def nearest_services():
    try:
        user_lat = float(request.args.get("lat"))
        user_lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "lat and lon parameters are required inputs"}), 400

    radius   = float(request.args.get("radius",  30))
    category = request.args.get("category", "all").lower().strip()
    limit    = int(request.args.get("limit",  50))

    df = service_df.copy()

    # Filter by category 
    if category != "all":
        df = df[df["category"] == category]

    if df.empty:
        return jsonify([])

    # Compute distances vectorially
    df["_dist"] = df.apply(lambda r: haversine(user_lat, user_lon, r["lat"], r["lon"]), axis=1)

    # Filter inside bounding limits, sort by range nearest, slice maximum limit output
    nearby = df[df["_dist"] <= radius].sort_values("_dist").head(limit)

    # Fallback default: if absolutely nothing is within requested range, return closest 20 elements globally
    if nearby.empty:
        nearby = df.sort_values("_dist").head(20)

    return jsonify([row_to_dict(row, row["_dist"]) for _, row in nearby.iterrows()])


@app.route("/api/search")
def search():
    query    = request.args.get("q", "").strip().lower()
    user_lat = request.args.get("lat", type=float)
    user_lon = request.args.get("lon", type=float)

    if not query:
        return jsonify([])

    df   = service_df.copy()
    mask = (
        df["name"].astype(str).str.lower().str.contains(query, na=False)     |
        df["address"].astype(str).str.lower().str.contains(query, na=False)  |
        df["city"].astype(str).str.lower().str.contains(query, na=False)     |
        df["services_offered"].astype(str).str.lower().str.contains(query, na=False)
    )
    result = df[mask].head(30).copy()

    if user_lat and user_lon:
        result["_dist"] = result.apply(lambda r: haversine(user_lat, user_lon, r["lat"], r["lon"]), axis=1)
        result = result.sort_values("_dist")
    else:
        result["_dist"] = 0.0

    return jsonify([row_to_dict(row, row["_dist"]) for _, row in result.iterrows()])


@app.route("/api/sos", methods=["POST"])
def sos():
    data = request.get_json(silent=True) or {}
    print(f"🚨 SOS ALARM TRACE  lat={data.get('lat')}  lon={data.get('lon')}  msg={data.get('message','')}")
    return jsonify({"status": "received", "message": "Emergency broadcast captured. Help is on the way"})


@app.route("/api/emergency")
def emergency():
    return jsonify({
        "ambulance":          "108",
        "police":             "100",
        "fire":               "101",
        "women_helpline":     "1091",
        "road_accident":      "1033",
        "disaster_mgmt":      "108",
        "national_emergency": "112",
    })


@app.route("/api/status")
def status():
    return jsonify({
        "status":      "running",
        "total":       len(service_df),
        "by_category": service_df["category"].value_counts().to_dict(),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)