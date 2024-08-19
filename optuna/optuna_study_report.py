import optuna
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_study_report(study_name, storage):
    try:
        study = optuna.load_study(
            study_name=study_name, 
            storage=storage
        )
        
        logger.info("Study report:")
        logger.info(f"Study name: {study.study_name}")
        logger.info(f"Direction: {study.direction}")
        
        # Print sampler and pruner information
        logger.info(f"Sampler: {study.sampler}")
        logger.info(f"Pruner: {study.pruner}")
        
        # Print some information about trials
        logger.info(f"Number of trials: {len(study.trials)}")
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        logger.info(f"Number of completed trials: {completed_trials}")
        
        if study.trials:
            logger.info("First trial parameters:")
            for key, value in study.trials[0].params.items():
                logger.info(f"  {key}: {value}")
                
            logger.info("First trial intermediate values:")
            for key, value in study.trials[0].intermediate_values.items():
                logger.info(f"  Step {key}: {value}")
        
    except Exception as e:
        logger.error(f"Error loading the study: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-name', type=str, required=True, help="Name of the Optuna study")
    parser.add_argument('--storage', type=str, required=True, help="Storage URL of the Optuna study")
    args = parser.parse_args()

    print_study_report(args.study_name, args.storage)
