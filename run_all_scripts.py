"""
MPHY0047 Coursework 1 - Run All Analysis Scripts
Execute this file to run all question scripts in sequence.

Usage:
    python run_all_scripts.py
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script:  {script_name}")
    print('='*60)

    result = subprocess.run([sys.executable, script_name], cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed with return code {result.returncode}")
        return False
    return True


def main():
    print("="*60)
    print("MPHY0047 Coursework 1 - Running All Analysis Scripts")
    print("="*60)

    scripts = [
        ("question1.py", "Question 1: Descriptive Statistics (+ 12 figures)"),
        ("question2.py", "Question 2: Statistical Testing (Time Parameters)"),
        ("question3.py", "Question 3: Error Analysis"),
        ("question4.py", "Question 4: Fixation Sparsity Analysis"),
        ("question5.py", "Question 5: Metric Ranking"),
    ]

    for i, (script, desc) in enumerate(scripts, 1):
        print(f"\n[{i}/{len(scripts)}] {desc}")
        if not run_script(script, desc):
            print(f"\nExecution stopped due to error in {script}")
            sys.exit(1)

    print("\n" + "="*60)
    print("All scripts completed successfully!")
    print("Figures saved to: figures/")
    print("="*60)


if __name__ == "__main__":
    main()
