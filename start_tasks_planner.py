from runs.planner_presets import run_tasks_planner, online_experiment

if __name__ == '__main__':
    # Start a batch of runs with pre-defined settings
    # run_tasks_planner(train_online_experiment, skip_existing_runs=True, tuning_task=False)
    run_tasks_planner(online_experiment, skip_existing_runs=True, tuning_task=True, online_tuning=True)
