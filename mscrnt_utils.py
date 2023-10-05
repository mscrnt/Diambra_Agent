import os

# Generate run_id
class RunIDGenerator:
    @classmethod
    def create(cls, settings, wrappers_settings, ppo_settings):
        run_id = f"SR_{settings['step_ratio']}-SA_{wrappers_settings['stack_actions']}"
        if wrappers_settings['normalize_reward']:
            run_id += f"-NF_{wrappers_settings['normalization_factor']:.1e}"
        
        return run_id


def generate_parameters_report(run_id, settings, wrappers_settings, hparams):
    log_dir = f"logs/{run_id}"
    os.makedirs(log_dir, exist_ok=True)
    report = f"""
    # Run ID: {run_id}
    # Settings: {settings}
    # Wrappers Settings: {wrappers_settings}
    # Hyperparameters: {hparams}
    """
    with open(f"{log_dir}/parameters.txt", "w") as f:
        f.write(report)
    return report
