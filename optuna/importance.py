import optuna
from optuna.importance import get_param_importances
from optuna.importance import FanovaImportanceEvaluator

# Load your existing study
study = optuna.load_study(study_name="ppo_study", storage="postgresql://optuna_user:password@localhost/optuna_db")

# Get hyperparameter importances
param_importances = get_param_importances(study, evaluator=FanovaImportanceEvaluator())

# Print the importances
for param, importance in param_importances.items():
    print(f"{param}: {importance:.3f}")
