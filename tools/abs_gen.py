import argparse
import csv
import sys
import os
import matplotlib.pyplot as plt

# Add src module to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from sharma_abs import SharmaABS

def main():
    parser = argparse.ArgumentParser(description="Sharma ABS Model CLI Generator")
    parser.add_argument("input_csv", help="Input CSV file with deterministic data")
    parser.add_argument("output_csv", help="Output CSV file for noisy data")
    parser.add_argument("--min", type=int, default=-10, help="Min boundary (default: -10)")
    parser.add_argument("--max", type=int, default=10, help="Max boundary (default: 10)")
    parser.add_argument("--complexity", "-n", type=int, default=10, help="Complexity / samples per cycle (default: 10)")
    parser.add_argument("--alpha", "-a", type=float, default=0.5, help="Influence coefficient, 0 to 1 (default: 0.5)")
    parser.add_argument("--col", type=int, default=1, help="0-indexed column of data to apply noise to (default: 1)")
    parser.add_argument("--plot", action="store_true", help="Generate a PNG plot showcasing the comparison")
    
    args = parser.parse_args()
    
    abs_model = SharmaABS(args.min, args.max, args.complexity)
    
    ideal_data = []
    headers = []
    all_rows = []
    
    try:
        with open(args.input_csv, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                all_rows.append(row)
                try:
                    val = float(row[args.col])
                    ideal_data.append(val)
                except ValueError:
                    print(f"Error: Non-numeric value '{row[args.col]}' in column {args.col}.")
                    return
    except Exception as e:
        print(f"Error reading {args.input_csv}: {e}")
        return

    print(f"Generating noisy dataset based on {len(ideal_data)} rows...")
    noisy_data = abs_model.generate_noisy_dataset(ideal_data, args.alpha)
    
    try:
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            # Add new header for noisy data
            out_headers = list(headers)
            out_headers.append(headers[args.col] + "_abs_obs")
            writer.writerow(out_headers)
            
            # Write rows
            for i, row in enumerate(all_rows):
                out_row = list(row)
                out_row.append(str(noisy_data[i]))
                writer.writerow(out_row)
        print(f"Successfully wrote {args.output_csv}")
    except Exception as e:
        print(f"Error writing {args.output_csv}: {e}")
        return
        
    if args.plot:
        plt.figure(figsize=(10, 6))
        plt.plot(ideal_data, label="Deterministic Ideal", linestyle='--', color='emerald')
        plt.plot(noisy_data, label=f"Sharma ABS Obs (alpha={args.alpha}, n={args.complexity})", color='indigo', alpha=0.9)
        plt.title("Sharma Absolute Boundary Summation (ABS) Model")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_file = os.path.splitext(args.output_csv)[0] + '.png'
        plt.savefig(plot_file)
        print(f"Plot correctly saved to {plot_file}")

if __name__ == "__main__":
    main()
