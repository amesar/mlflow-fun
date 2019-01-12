from __future__ import print_function
import sys
from wine_quality import train_wine_quality

def run(data_path, x, y, run_origin):
    train_wine_quality.train(data_path, x,y, run_origin)

if __name__ == "__main__":
    if len(sys.argv) < 3: 
        print("ERROR: Expecting alpha and l1_ratio values")
        sys.exit(1)
    data_path = sys.argv[3] if len(sys.argv) > 3 else "wine-quality.csv"
    run_origin = sys.argv[4] if len(sys.argv) > 4 else ""
    run(data_path, float(sys.argv[1]), float(sys.argv[2]), run_origin)
