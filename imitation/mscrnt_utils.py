import os

# Generate run_id
class RunIDGenerator:
    """
    A class for generating unique run IDs based on various settings.

    Attributes:
    -----------
    None

    Methods:
    --------
    create(cls, settings, wrappers_settings, ppo_settings):
        Generates a unique run ID based on the provided settings.

    """
    @classmethod
    def create(cls, settings, wrappers_settings, ppo_settings):


        run_id = f"SR{settings['step_ratio']}-SA{wrappers_settings['stack_actions']}"
        if wrappers_settings['normalize_reward']:
            run_id += f"-NF{wrappers_settings['normalization_factor']}"
        
        return run_id


def generate_parameters_report(run_id, settings, wrappers_settings, hparams, log_dir):
    report = f"""
    # Run ID: {run_id}
    # Settings: {settings}
    # Wrappers Settings: {wrappers_settings}
    # Hyperparameters: {hparams}
    """
    with open(f"{log_dir}/parameters.txt", "w") as f:
        f.write(report)
    return report
