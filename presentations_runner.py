import os
import glob
import importlib.util
import re
import pandas as pd
import typer
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(ROOT, "examples")

FILES = {
    "basic": os.path.join(EXAMPLES_DIR, "01_presentation_basic_results_no_color.py"),
    "performance": os.path.join(EXAMPLES_DIR, "02_presentation_performance_summary_no_color.py"),
    "errors": os.path.join(EXAMPLES_DIR, "03_presentation_error_analysis_no_color.py"),
    "comparison": os.path.join(EXAMPLES_DIR, "04_presentation_comparison_table.py"),
    "summary": os.path.join(EXAMPLES_DIR, "05_presentation_summary_no_color.py"),
}

def _resolve_csv_path(csv_path: str | None) -> str:
    """Return provided csv_path if exists; otherwise pick latest in reports/*/data/ocr_evaluation_detailed.csv"""
    if csv_path and os.path.exists(csv_path):
        return csv_path
    candidates = sorted(glob.glob(os.path.join(ROOT, "reports", "*", "data", "ocr_evaluation_detailed.csv")))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"CSV not found. Provided: {csv_path!r}. Also searched: reports/*/data/ocr_evaluation_detailed.csv")


def _load_module(key: str):
    """Dynamically load a module from examples by key in FILES."""
    path = FILES[key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module file not found: {path}")
    mod_name = f"examples_{re.sub(r'[^0-9a-zA-Z_]+', '_', os.path.basename(path))}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

app = typer.Typer(help="Unified CLI to run presentation functions from examples.")

@app.command()
def basic(
    csv: Optional[str] = typer.Option(None, "--csv", help="Path to ocr_evaluation_detailed.csv"),
    min_examples: int = typer.Option(1, "--min", help="Number of MIN CER examples to show"),
    max_examples: int = typer.Option(1, "--max", help="Number of MAX CER examples to show"),
    mean_examples: int = typer.Option(1, "--mean", help="Number of MEAN CER examples to show"),
    export_csv: Optional[str] = typer.Option(None, "--export", help="Export results to CSV file"),
    mode: str = typer.Option("standard", "--mode", help="Display mode: standard, with-names, by-file")
):
    """Run basic results with configurable display modes and examples."""
    module = _load_module("basic")
    csv_path = _resolve_csv_path(csv)
    
    if mode == "standard":
        module.present_basic_results(csv_path=csv_path, min_examples=min_examples, max_examples=max_examples, mean_examples=mean_examples, export_csv=export_csv)
    elif mode == "with-names":
        module.present_basic_results_with_names(csv_path=csv_path, min_examples=min_examples, max_examples=max_examples, mean_examples=mean_examples, export_csv=export_csv)
    elif mode == "by-file":
        module.present_results_by_file(csv_path=csv_path, min_examples=min_examples, max_examples=max_examples, mean_examples=mean_examples, export_csv=export_csv)
    else:
        raise typer.BadParameter(f"Invalid mode: {mode}. Choose from: standard, with-names, by-file")

@app.command("performance-summary")
def performance_summary(csv: Optional[str] = typer.Option(None, "--csv", help="Path to detailed CSV")):
    """Run performance summary and show best performers."""
    module = _load_module("performance")
    csv_path = _resolve_csv_path(csv)
    df = pd.read_csv(csv_path)
    module.present_performance_summary(csv_path=csv_path)
    module.find_best_performers(df)

@app.command("error-analysis")
def error_analysis(
    csv: Optional[str] = typer.Option(None, "--csv", help="Path to detailed CSV"),
    export_csv: Optional[str] = typer.Option(None, "--export", help="Export worst cases to CSV file")
):
    """Run error analysis with CSV export option."""
    module = _load_module("errors")
    csv_path = _resolve_csv_path(csv)
    module.present_error_analysis(csv_path=csv_path, export_csv=export_csv)

@app.command("error-patterns")
def error_patterns(
    csv: Optional[str] = typer.Option(None, "--csv", help="Path to detailed CSV"),
    export_csv: Optional[str] = typer.Option(None, "--export", help="Export error patterns to CSV file")
):
    """Show common error patterns with CSV export option."""
    module = _load_module("errors")
    csv_path = _resolve_csv_path(csv)
    module.present_error_patterns(csv_path=csv_path, export_csv=export_csv)
    csv_path = _resolve_csv_path(csv)
    module.present_error_patterns(csv_path=csv_path)


@app.command("error-distribution")
def error_distribution(
    csv: Optional[str] = typer.Option(None, "--csv", help="Path to detailed CSV"),
    export_csv: Optional[str] = typer.Option(None, "--export", help="Export error distribution to CSV file")
):
    """Show error distribution ranges with CSV export option."""
    module = _load_module("errors")
    csv_path = _resolve_csv_path(csv)
    df = pd.read_csv(csv_path)
    module.analyze_error_distribution(df, export_csv=export_csv)

@app.command("accuracy-distribution")
def accuracy_distribution(
    csv: Optional[str] = typer.Option(None, "--csv", help="Path to detailed CSV")
):
    """Show accuracy distribution for each model-dataset combination."""
    module = _load_module("performance")
    csv_path = _resolve_csv_path(csv)
    module.present_accuracy_distribution(csv_path)

@app.command("comparison-best")
def comparison_best(csv: Optional[str] = typer.Option(None, "--csv", help="Path to detailed CSV")):
    """Show best model per dataset using comparison script (colored)."""
    module = _load_module("comparison")
    csv_path = _resolve_csv_path(csv)
    df = pd.read_csv(csv_path)
    module.find_best_performers(df)

@app.command()
def all(csv: Optional[str] = typer.Option(None, "--csv", help="Path to detailed CSV")):
    """Run a compact sequence of useful views."""
    csv_path = _resolve_csv_path(csv)
    basic(csv_path)
    performance_summary(csv_path)
    error_analysis(csv_path)
    error_patterns(csv_path)
    error_distribution(csv_path)
    comparison_best(csv_path)

if __name__ == "__main__":
    app()