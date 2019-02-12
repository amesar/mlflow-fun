from __future__ import print_function
import sys
from wine_quality import train_wine_quality

if __name__ == "__main__":
    if len(sys.argv) < 4: 
        print("ERROR: Expecting EXPERIMENT_NAME, ALPHA, L1_RATIO and DATA_FILE")
        print("  Arguments: ",sys.argv)
        sys.exit(1)
    data_path = sys.argv[4] if len(sys.argv) > 4 else "wine-quality.csv"
    run_origin = sys.argv[5] if len(sys.argv) > 5 else "none"
    train_wine_quality.train(sys.argv[1], data_path, float(sys.argv[2]), float(sys.argv[3]), run_origin)
