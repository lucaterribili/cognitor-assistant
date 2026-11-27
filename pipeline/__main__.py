import argparse
from pipeline import run_full_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline di addestramento Cognitor")
    parser.add_argument(
        "--no-policy",
        "--skip-policy",
        action="store_true",
        dest="no_policy",
        help="Esegue la pipeline fino all'Intent Classifier (salta la Dialogue Policy)",
    )
    parser.add_argument(
        "--no-classifier",
        "--skip-classifier",
        action="store_true",
        dest="no_classifier",
        help="Salta il training dell'Intent Classifier",
    )
    args = parser.parse_args()

    run_full_pipeline(
        train_classifier=not args.no_classifier,
        train_policy=not args.no_policy,
    )

