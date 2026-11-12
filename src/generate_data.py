from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RANDOM_SEED = 42


def generate_claims(n_rows: int = 450, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    payers = ["Aetna", "Cigna", "UnitedHC", "BlueCross", "Humana", "Kaiser"]
    providers = ["Dr.Rao", "Dr.Shah", "Dr.Lee", "Dr.Patel", "Dr.Nguyen", "Dr.Mitchell"]
    cpt_codes = ["99213", "99214", "99215", "93000", "36415", "93010", "27447", "90834"]
    diagnosis_codes = ["M54.5", "J06.9", "E11.9", "K35.2", "S72.0", "I10", "N39.0", "R07.9"]

    rows = []
    statuses = np.random.choice(
        ["Paid", "Partially_Paid", "Denied"],
        size=n_rows,
        p=[0.7, 0.2, 0.1],
        replace=True,
    )

    for i, status in enumerate(statuses, start=1):
        payer = rng.choice(payers)
        provider = rng.choice(providers)
        cpt = rng.choice(cpt_codes)
        diagnosis = rng.choice(diagnosis_codes)
        patient_age = int(rng.integers(18, 90))

        if status == "Paid":
            billed_amount = float(rng.uniform(120, 650))
        elif status == "Partially_Paid":
            billed_amount = float(rng.uniform(180, 900))
        else:
            billed_amount = float(rng.uniform(200, 950))

        payer_bias = {
            "Aetna": {"Paid": 0.96, "Partially_Paid": 0.98, "Denied": 1.18},
            "Cigna": {"Paid": 0.92, "Partially_Paid": 0.88, "Denied": 1.12},
            "UnitedHC": {"Paid": 0.91, "Partially_Paid": 1.02, "Denied": 1.2},
            "BlueCross": {"Paid": 0.94, "Partially_Paid": 1.0, "Denied": 1.16},
            "Humana": {"Paid": 0.95, "Partially_Paid": 1.05, "Denied": 1.14},
            "Kaiser": {"Paid": 0.9, "Partially_Paid": 1.0, "Denied": 1.28},
        }
        billed_amount *= payer_bias[payer][status]

        rows.append(
            {
                "id": i,
                "payer": payer,
                "provider": provider,
                "cpt_code": cpt,
                "diagnosis_code": diagnosis,
                "billed_amount": round(float(billed_amount), 2),
                "patient_age": patient_age,
                "status": status,
            }
        )

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_claims()
    path = DATA_DIR / "claims.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    print(df["status"].value_counts(normalize=True).to_dict())


if __name__ == "__main__":
    main()
