import subprocess
import sys

def run_command(command):
    print("\n" + "=" * 70)
    print("Running:", " ".join(command))
    print("=" * 70)

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\nCommand failed with exit code {result.returncode}.")
        sys.exit(result.returncode)

def main():

    python = sys.executable

    commands = [
        [
            python,
            "-m",
            "src.preprocessing.preprocess_pipeline",
            "--dataset",
            "all",
        ],
        [
            python,
            "-m",
            "src.experiments.experiment1_similarity",
        ],
        [
            python,
            "-m",
            "src.experiments.experiment2_classifier",
        ],
        [
            python,
            "-m",
            "src.experiments.experiment3_finetune",
        ],
        [
            python,
            "-m",
            "notebooks.plot_experiment1_results",
        ],
        [
            python,
            "-m",
            "notebooks.plot_experiment2_results",
        ],
        [
            python,
            "-m",
            "notebooks.plot_experiment3_results",
        ],
    ]

    for command in commands:
        run_command(command)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()