#!/usr/bin/env python3
"""
BPM Prediction Project - Main Entry Point

This script provides easy access to all project functionality.
Run with different commands to execute various components.

Usage:
    python main.py --help                    # Show help
    python main.py --run-all                 # Run complete pipeline
    python main.py --experimental            # Run experimental features only
    python main.py --pipeline               # Run main pipeline only
    python main.py --evaluate               # Run evaluation only
    python main.py --summary                # Show project summary
"""

import argparse
import sys
import os
from pathlib import Path

# Add src and scripts to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'scripts'))

def run_experimental():
    """Run experimental feature engineering"""
    print("🔬 Running experimental feature engineering...")
    from scripts.experimental_approaches import run_experimental_pipeline
    train_exp, test_exp = run_experimental_pipeline()
    print(f"✅ Experimental features created: {train_exp.shape[1]} features")
    return train_exp, test_exp

def run_pipeline():
    """Run main modeling pipeline"""
    print("🚀 Running main modeling pipeline...")
    from scripts.run_pipeline import main
    submission = main()
    print(f"✅ Pipeline completed: {len(submission)} predictions generated")
    return submission

def run_evaluation():
    """Run complete evaluation"""
    print("📊 Running complete evaluation...")
    from scripts.run_complete_evaluation import main
    results = main()
    print("✅ Evaluation completed")
    return results

def show_summary():
    """Show project summary"""
    print("📄 Showing project summary...")
    from scripts.project_summary import main
    main()

def run_all():
    """Run complete pipeline"""
    print("🎵 Running complete BPM prediction pipeline...")
    print("=" * 60)
    
    # Run all components
    run_experimental()
    run_pipeline()
    run_evaluation()
    show_summary()
    
    print("\n🎉 Complete pipeline finished!")

def main():
    parser = argparse.ArgumentParser(
        description="BPM Prediction Project - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --run-all          # Run everything
  python main.py --experimental     # Just feature engineering
  python main.py --pipeline         # Just modeling
  python main.py --evaluate         # Just evaluation
  python main.py --summary          # Just summary
        """
    )
    
    parser.add_argument('--run-all', action='store_true', 
                       help='Run complete pipeline (recommended)')
    parser.add_argument('--experimental', action='store_true',
                       help='Run experimental feature engineering only')
    parser.add_argument('--pipeline', action='store_true',
                       help='Run main modeling pipeline only')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation and analysis only')
    parser.add_argument('--summary', action='store_true',
                       help='Show project summary only')
    
    args = parser.parse_args()
    
    # Check if no arguments provided
    if not any([args.run_all, args.experimental, args.pipeline, 
                args.evaluate, args.summary]):
        parser.print_help()
        return
    
    # Execute requested components
    try:
        if args.run_all:
            run_all()
        elif args.experimental:
            run_experimental()
        elif args.pipeline:
            run_pipeline()
        elif args.evaluate:
            run_evaluation()
        elif args.summary:
            show_summary()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check that all dependencies are installed and data files exist")
        sys.exit(1)

if __name__ == "__main__":
    main()
