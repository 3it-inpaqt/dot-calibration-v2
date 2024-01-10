from runs.planner_presets import full_scan_all, full_scan_cross_valid, run_tasks_planner, tune_all, \
    tune_all_cross_valid, tune_oracle

# Start a batch of runs with pre-defined settings to get every result for a specific seed.
# It is possible to run multiple instances of this script in parallel with different seeds.
# Every existing run will be skipped.
if __name__ == '__main__':
    # Run tuning with oracle and random as a baseline
    run_tasks_planner(tune_oracle)

    # =================== Mixed Diagrams ===================

    # Training and tuning
    run_tasks_planner(tune_all)

    # Full scan for each diagram for the first seed
    run_tasks_planner(full_scan_all)

    # Uncertainty study
    # run_tasks_planner(uncertainty_study_all, tuning_task=False)

    # ================== Cross-Validation ==================

    # Training and tuning
    run_tasks_planner(tune_all_cross_valid)

    # Full scan for each diagram for the first seed
    run_tasks_planner(full_scan_cross_valid)

    # Uncertainty study
    # run_tasks_planner(uncertainty_study_cross_valid, tuning_task=False)
