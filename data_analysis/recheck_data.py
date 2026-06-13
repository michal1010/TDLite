from csv import DictWriter
from pathlib import Path

from experiments.run_humaneval import evaluate_solution
from experiments.humaneval_loader import get_hidden_tests

def recheck_data(scenario: str, subfolder: str, start_instance: int = 0, end_instance: int = 164) -> None:
  repo_root = Path(__file__).resolve().parents[1]
  scenario_path = Path(scenario)
  if not scenario_path.is_absolute():
    scenario_path = repo_root / "data_analysis" / scenario_path

  base_dir = (scenario_path / subfolder).resolve()
  output_path = base_dir / "pass_per_instance.csv"
  if output_path.exists():
    return

  print(base_dir)

  experiment_files_dir = base_dir / "experiment_files"
  hidden_tests = get_hidden_tests()

  results: list[dict[str, object]] = []
  for instance, tests in sorted(hidden_tests.items()):
    if not start_instance <= instance <= end_instance:
      continue
    generated_program_path = experiment_files_dir / f"instance_{instance - start_instance + 1}" / "generated_program.py"
    try: 
      code = generated_program_path.read_text(encoding="utf-8")
    except FileNotFoundError:
      print(f"File for instance{instance + 1} not found (could be normal behavior).")
      break
    passed, _ = evaluate_solution(code, tests["test"], tests["entry_point"])
    results.append({"instance_id": instance, "passed": passed})

  base_dir.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", newline="", encoding="utf-8") as handle:
    writer = DictWriter(handle, fieldnames=["instance_id", "passed"])
    writer.writeheader()
    writer.writerows(results)