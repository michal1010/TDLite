source .venv/Scripts/activate # Or wherever your venv lives

DATA_ROOT="${1:-data_analysis}"

declare -A SCENARIOS=(
  ["default"]="default_0_54 default_55_109 default_110_163"
  ["refine"]="REF_0_54 REF_55_109 REF_110_163"
  ["tdd"]="tdd_0_50 tdd_51_105 tdd_106_163"
  ["tdd_refine"]="TDD_REF_0_54 TDD_REF_55_109 TDD_REF_110_163"
)

total=0      # instances seen
passed=0     # tests exited 0
failed=0     # tests exited non-zero (still coverage-measured)
skipped=0    # missing folders/files
 
for prefix in "${!SCENARIOS[@]}"; do
  for subfolder in ${SCENARIOS[$prefix]}; do
 
    exp_dir="$DATA_ROOT/$prefix/$subfolder/experiment_files"
 
    if [[ ! -d "$exp_dir" ]]; then
      echo "WARNING: missing experiment dir, skipping: $exp_dir" >&2
      continue
    fi
 
    for instance_dir in "$exp_dir"/instance_*/; do
      [[ -d "$instance_dir" ]] || continue   # no matches -> skip the literal glob
      total=$((total + 1))
 
      echo "=== $instance_dir ==="
 
      # Run in a subshell: the cd and any failure stay contained here,
      # so the outer loop always returns to the original working directory.
      (
        cd "$instance_dir" || { echo "  ERROR: cannot cd into folder" >&2; exit 2; }
 
        if [[ ! -f generated_program.py ]]; then
          echo "  WARNING: generated_program.py not found, skipping" >&2
          exit 3
        fi
 
        # 2. copy the program into an outputs/ folder
        mkdir -p outputs
        cp generated_program.py outputs/
 
        # 3. branch coverage over the unit tests
        coverage run --branch -m unittest generated_tests
        run_status=$?            # exit code of the test run (failing tests -> non-zero)
 
        # 4. emit the JSON report (still works even if some tests failed)
        coverage json
 
        # Surface the *test-run* status as this subshell's exit code.
        exit "$run_status"
      )
      status=$?
 
      case "$status" in
        0) passed=$((passed + 1)) ;;
        3) skipped=$((skipped + 1)); total=$((total - 1)) ;;   # nothing to run here
        *) failed=$((failed + 1))
           echo "  NOTE: tests exited non-zero (coverage was still recorded)" ;;
      esac
 
    done
  done
done
 
# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
 
echo
echo "------------------------------------------------------------"
echo "Done. Instances processed: $total"
echo "  tests passed : $passed"
echo "  tests failed : $failed   (coverage.json still generated)"
echo "  skipped      : $skipped  (missing generated_program.py)"
echo "------------------------------------------------------------"
