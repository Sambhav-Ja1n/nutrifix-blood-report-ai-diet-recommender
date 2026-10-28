
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nutrifix.data_quality import main as run_data_quality
from src.nutrifix.preprocessing import main as run_preprocessing
from src.nutrifix.anomaly_detection import main as run_anomaly_detection


def main():
    print("\n" + "=" * 60)
    print("STEP 1/3: DATA QUALITY")
    print("=" * 60)
    run_data_quality()

    print("\n" + "=" * 60)
    print("STEP 2/3: FEATURE PREPROCESSING")
    print("=" * 60)
    run_preprocessing()

    print("\n" + "=" * 60)
    print("STEP 3/3: ANOMALY DETECTION")
    print("=" * 60)
    run_anomaly_detection()

    print("\nNutriFix pipeline completed successfully.")


if __name__ == "__main__":
    main()
