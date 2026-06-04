import pandas as pd
import os

CSV_PATH = "dataset/indonesian_food/nutrition.csv"

# Predefined mapping for the 13 CNN classification labels to entries in nutrition.csv
# or manually specified nutritional values when absent.
# Values are per 100g.
CLASS_MAPPING = {
    "ayam goreng": {
        "db_name": "Ayam goreng paha",  # ID 40
        "fallback": {
            "calories": 287.0,
            "proteins": 31.0,
            "fat": 15.7,
            "carbohydrate": 1.7,
            "fiber": 0.0,
            "sugar": 0.0,
            "sodium": 0.0
        }
    },
    "burger": {
        "db_name": "Beef burger",  # ID 91
        "fallback": {
            "calories": 258.0,
            "proteins": 10.6,
            "fat": 9.5,
            "carbohydrate": 32.5,
            "fiber": 1.6,
            "sugar": 5.0,
            "sodium": 458.0
        }
    },
    "french fries": {
        "db_name": None,  # Absent in database, using custom manual fallback
        "fallback": {
            "calories": 312.0,
            "proteins": 3.4,
            "fat": 15.0,
            "carbohydrate": 41.0,
            "fiber": 3.8,
            "sugar": 0.3,
            "sodium": 210.0
        }
    },
    "gado-gado": {
        "db_name": "Gado-gado",  # ID 356
        "fallback": {
            "calories": 137.0,
            "proteins": 4.8,
            "fat": 4.8,
            "carbohydrate": 18.7,
            "fiber": 2.0,
            "sugar": 3.0,
            "sodium": 280.0
        }
    },
    "ikan goreng": {
        "db_name": "Ikan Mas goreng",  # ID 474
        "fallback": {
            "calories": 188.0,
            "proteins": 19.3,
            "fat": 12.2,
            "carbohydrate": 0.0,
            "fiber": 0.0,
            "sugar": 0.0,
            "sodium": 80.0
        }
    },
    "mie goreng": {
        "db_name": "Mie Goreng  ",  # ID 898 (note trailing spaces in CSV)
        "fallback": {
            "calories": 468.0,
            "proteins": 10.4,
            "fat": 22.8,
            "carbohydrate": 55.4,
            "fiber": 2.0,
            "sugar": 4.0,
            "sodium": 920.0
        }
    },
    "nasi goreng": {
        "db_name": "Nasi Goreng  ",  # ID 924 (note trailing spaces in CSV)
        "fallback": {
            "calories": 276.0,
            "proteins": 5.5,
            "fat": 8.7,
            "carbohydrate": 44.0,
            "fiber": 1.2,
            "sugar": 1.5,
            "sodium": 580.0
        }
    },
    "nasi padang": {
        "db_name": "Nasi rames",  # ID 928 (closest equivalent)
        "fallback": {
            "calories": 155.0,
            "proteins": 3.6,
            "fat": 4.6,
            "carbohydrate": 24.8,
            "fiber": 1.5,
            "sugar": 1.0,
            "sodium": 340.0
        }
    },
    "pizza": {
        "db_name": None,  # Absent in database, using custom manual fallback
        "fallback": {
            "calories": 266.0,
            "proteins": 11.4,
            "fat": 9.8,
            "carbohydrate": 33.0,
            "fiber": 2.3,
            "sugar": 3.6,
            "sodium": 556.0
        }
    },
    "rawon": {
        "db_name": "Rawon masakan",  # ID 1037
        "fallback": {
            "calories": 60.0,
            "proteins": 6.9,
            "fat": 2.9,
            "carbohydrate": 1.5,
            "fiber": 0.5,
            "sugar": 0.8,
            "sodium": 390.0
        }
    },
    "rendang": {
        "db_name": "Rendang sapi masakan",  # ID 1045
        "fallback": {
            "calories": 193.0,
            "proteins": 22.6,
            "fat": 7.9,
            "carbohydrate": 7.8,
            "fiber": 1.0,
            "sugar": 1.2,
            "sodium": 420.0
        }
    },
    "sate": {
        "db_name": "Sate pusut masakan",  # ID 1103
        "fallback": {
            "calories": 274.0,
            "proteins": 2.2,
            "fat": 2.2,
            "carbohydrate": 19.6,
            "fiber": 0.5,
            "sugar": 1.5,
            "sodium": 310.0
        }
    },
    "soto": {
        "db_name": "Soto dengan Daging",  # ID 1155
        "fallback": {
            "calories": 127.5,
            "proteins": 6.8,
            "fat": 7.9,
            "carbohydrate": 7.2,
            "fiber": 0.4,
            "sugar": 0.6,
            "sodium": 410.0
        }
    }
}


def estimate_weight(area):
    """
    Estimasi berat berdasarkan luas contour
    """
    if area <= 0:
        return 0

    weight = (area / 1000) * 6
    return round(weight, 2)


def get_nutrition_data(food_name, weight):
    """
    Ambil data nutrisi dari nutrition.csv dengan mapping dan fallback
    """
    default_data = {
        "calories": 0,
        "protein": 0,
        "fat": 0,
        "carbohydrate": 0,
        "fiber": 0,
        "sugar": 0,
        "sodium": 0
    }

    if not food_name:
        return default_data

    # Normalize input name (lowercase, strip, replace underscores)
    food_name_clean = food_name.lower().strip().replace("_", " ")

    target_db_name = None
    fallback_values = None

    # Step 1: Check in the translation dictionary
    if food_name_clean in CLASS_MAPPING:
        target_db_name = CLASS_MAPPING[food_name_clean]["db_name"]
        fallback_values = CLASS_MAPPING[food_name_clean]["fallback"]
    else:
        # Check if the query is a substring of any keys
        for key, val in CLASS_MAPPING.items():
            if key in food_name_clean or food_name_clean in key:
                target_db_name = val["db_name"]
                fallback_values = val["fallback"]
                break

    # Load nutrition database
    df = None
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            # Normalize CSV column names to prevent casing issues
            df.columns = [col.lower().strip() for col in df.columns]
        except Exception as e:
            print(f"Error reading CSV {CSV_PATH}: {e}")

    row = None

    if df is not None:
        # Match using target name from dictionary
        if target_db_name:
            matches = df[df['name'].astype(str).str.lower().str.strip() == target_db_name.lower().strip()]
            if not matches.empty:
                row = matches.iloc[0]

        # Substring/fuzzy match fallback
        if row is None:
            matches = df[df['name'].astype(str).str.lower().str.contains(food_name_clean, regex=False)]
            if not matches.empty:
                row = matches.iloc[0]
            else:
                # Try parts of the name (minimum word length 3)
                parts = [p for p in food_name_clean.split() if len(p) >= 3]
                for part in parts:
                    matches = df[df['name'].astype(str).str.lower().str.contains(part, regex=False)]
                    if not matches.empty:
                        row = matches.iloc[0]
                        break

    # Get values from row or fallback values
    if row is not None:
        def safe_value(col_name):
            try:
                val = row[col_name]
                return float(val) if pd.notna(val) else 0.0
            except:
                return 0.0

        calories_base = safe_value('calories')
        protein_base = safe_value('proteins')
        fat_base = safe_value('fat')
        carb_base = safe_value('carbohydrate')

        # Since CSV might not contain fiber, sugar, sodium columns, fallback to predefined values
        fiber_base = safe_value('fiber')
        if fiber_base == 0.0 and fallback_values:
            fiber_base = fallback_values.get('fiber', 0.0)

        sugar_base = safe_value('sugar')
        if sugar_base == 0.0 and fallback_values:
            sugar_base = fallback_values.get('sugar', 0.0)

        sodium_base = safe_value('sodium')
        if sodium_base == 0.0 and fallback_values:
            sodium_base = fallback_values.get('sodium', 0.0)
    elif fallback_values is not None:
        # No DB row found, use the manually-defined fallback metrics
        calories_base = fallback_values['calories']
        protein_base = fallback_values['proteins']
        fat_base = fallback_values['fat']
        carb_base = fallback_values['carbohydrate']
        fiber_base = fallback_values.get('fiber', 0.0)
        sugar_base = fallback_values.get('sugar', 0.0)
        sodium_base = fallback_values.get('sodium', 0.0)
    else:
        # Absolute fallback if neither DB nor mapping is found
        return default_data

    # Scale base values (metrics are per 100g)
    scale = weight / 100.0

    return {
        "calories": round(calories_base * scale, 2),
        "protein": round(protein_base * scale, 2),
        "fat": round(fat_base * scale, 2),
        "carbohydrate": round(carb_base * scale, 2),
        "fiber": round(fiber_base * scale, 2),
        "sugar": round(sugar_base * scale, 2),
        "sodium": round(sodium_base * scale, 2)
    }