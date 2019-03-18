from __future__ import print_function
import sys
from wine_quality import train_wine_quality

if __name__ == "__main__":
    if len(sys.argv) < 4: 
        print("ERROR: Expecting EXPERIMENT_NAME, DATA_PATH, ALPHA, L1_RATIO")
        print("  Arguments: ",sys.argv)
        sys.exit(1)
    run_origin = sys.argv[5] if len(sys.argv) > 5 else "none"
    train_wine_quality.train(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), run_origin)
