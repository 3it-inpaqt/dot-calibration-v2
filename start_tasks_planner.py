from runs.planner_presets import run_tasks_planner, tune_all_cross_valid

if __name__ == '__main__':
    # Start a batch of runs with pre-defined settings
    run_tasks_planner(tune_all_cross_valid, skip_existing_runs=True, tuning_task=True)
