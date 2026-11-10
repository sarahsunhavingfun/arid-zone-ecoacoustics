"""The species population behind every soundscape-driven comparison: species that BOTH
(a) occurred in the 28.67 h of annotated test soundscapes and (b) are an actual trained
BHC01 class.

Two ground-truth species were never trained as a BHC01 class and so cannot be scored by
any model, foundation or regional: 'Mole Cricket' (not prevalent enough to extract
training data) and 'Plague Locust' (could not be trained; see section 4.5). The usable
population is therefore 52 of the 54 species identified in the test soundscapes.

Source files (sarahdrive):
  - outdated/species_ranked_by_call_seconds.csv : the 54 ground-truth species (common names)
  - Hyperparameters and training stochasticity/BHC01_10repeat/bhc01_repeat1_Labels.txt :
    the 203-class label file

One spelling fix is needed: the label file says "Grey Teal" (Anas gracilis), the
ground-truth annotations say "Gray Teal" -- same species.
"""
import csv

FR = "/Volumes/sarahdrive/output/recognisers/final results"
LABELS = f"{FR}/Hyperparameters and training stochasticity/BHC01_10repeat/bhc01_repeat1_Labels.txt"
GT_CSV = f"{FR}/outdated/species_ranked_by_call_seconds.csv"

NOT_TRAINED = {"mole cricket", "plague locust"}


def _label_to_common(label: str) -> str:
    common = (label.split("_", 1)[1] if "_" in label else label).strip().lower()
    return "gray teal" if common == "grey teal" else common


def get_bhc01_classes() -> set:
    """All 203 trained BHC01 class names (common name, lowercase)."""
    classes = set()
    with open(LABELS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                classes.add(_label_to_common(line))
    return classes


def get_ground_truth_54() -> list:
    """The 54 species identified in the annotated test soundscapes (common name, lowercase)."""
    with open(GT_CSV, newline="", encoding="utf-8") as f:
        return [r["species"].strip().lower() for r in csv.DictReader(f)]


def get_population() -> list:
    """Ground-truth species that are also real BHC01 classes -- n = 52."""
    classes = get_bhc01_classes()
    gt = get_ground_truth_54()
    return sorted(s for s in gt if s in classes)


if __name__ == "__main__":
    gt = get_ground_truth_54()
    classes = get_bhc01_classes()
    population = sorted(s for s in gt if s in classes)
    excluded = sorted(s for s in gt if s not in classes)
    print(f"ground-truth species: {len(gt)}")
    print(f"usable population (real BHC01 classes): {len(population)}")
    print(f"excluded ({len(excluded)}): {excluded}")
    assert set(excluded) == NOT_TRAINED, "unexpected exclusion set -- check the label file"
