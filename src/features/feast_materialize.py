"""
Phase 1 — Feast Feature Store Materialization
Applies feature definitions and materializes offline → online store (Redis).

Run via DVC:  dvc repro featurize
Run directly: python src/features/feast_materialize.py

Prerequisite: Redis running (locally: docker run -p 6379:6379 redis:alpine)
              or on GKE (see k8s/feast/)
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path

from feast import FeatureStore

ROOT = Path(__file__).parent.parent.parent
FEATURE_REPO = ROOT / "src" / "features" / "feature_repo"
FEATURE_STORE_OUT = ROOT / "data" / "processed" / "feature_store"


def main() -> None:
    print(f"[featurize] Initializing Feast store from {FEATURE_REPO}")
    store = FeatureStore(repo_path=str(FEATURE_REPO))

    # Apply registers feature definitions to the Feast registry (GCS in prod)
    print("[featurize] Running feast apply ...")
    store.apply(objects=[], objects_to_delete=[], partial=False)

    # Materialize: push offline parquet → online Redis store
    # We materialize from the epoch to now, covering all training data
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = datetime.now(tz=timezone.utc)

    print(f"[featurize] Materializing {start_date.date()} → {end_date.date()} ...")
    store.materialize(start_date=start_date, end_date=end_date)

    # Copy registry snapshot for DVC tracking (so DVC can detect changes)
    FEATURE_STORE_OUT.mkdir(parents=True, exist_ok=True)
    registry_path = FEATURE_REPO / "data" / "registry.db"
    if registry_path.exists():
        shutil.copy(registry_path, FEATURE_STORE_OUT / "registry.db")

    print("[featurize] ✓ Feature store materialized and ready for online serving")
    print(f"[featurize] Test retrieval:")
    print("  from feast import FeatureStore")
    print(f"  store = FeatureStore(repo_path='{FEATURE_REPO}')")
    print("  features = store.get_online_features(")
    print("      features=['transaction_features:amount_log', ...],"  )
    print("      entity_rows=[{'transaction_id': 0}]")
    print("  )")


if __name__ == "__main__":
    main()
