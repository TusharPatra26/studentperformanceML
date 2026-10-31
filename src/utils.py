import os
import sys
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for model_name, model in models.items():
            model_params = param.get(model_name, {})
            if model_params:
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=model_params,
                    n_iter=min(10, len(ParameterGrid(model_params))),
                    scoring="r2",
                    cv=3,
                    random_state=42,
                    n_jobs=1,
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                models[model_name] = model
            else:
                model.fit(X_train, y_train)

            report[model_name] = r2_score(y_test, model.predict(X_test))

        return report
    except Exception as e:
        raise CustomException(e, sys)