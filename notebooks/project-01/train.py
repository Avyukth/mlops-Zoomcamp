import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from data_process import main as process_data


def create_base_models():
    return {
        'Logistic Regression': LogisticRegression(),
        'Support Vector Machine': SVC(kernel='linear'),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Neural Network': MLPClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Extra Trees': ExtraTreesClassifier(),
        'CatBoost': CatBoostClassifier(verbose=0),
        'LightGBM': LGBMClassifier(),
        'XGBoost': XGBClassifier(),
        'Decision Tree': DecisionTreeClassifier()
    }

# def perform_grid_search(x_train, y_train):
#     cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
#     pipelines = {
#         'Logistic': Pipeline([('MinMaxScale', MinMaxScaler()), ('Logistic', LogisticRegression())]),
#         'SVC': Pipeline([('MinMaxScale', MinMaxScaler()), ('SVC', SVC(probability=True))]),
#         'KNN': Pipeline([('MinMaxScale', MinMaxScaler()), ('KNN', KNeighborsClassifier())]),
#         'MLP': Pipeline([('MinMaxScale', MinMaxScaler()), ('MLP', MLPClassifier())]),
#         'RandomForest': Pipeline([('RandomForest', RandomForestClassifier())]),
#         'ExtraTrees': Pipeline([('ExtraTrees', ExtraTreesClassifier())]),
#         'CatBoost': Pipeline([('CatBoost', CatBoostClassifier(verbose=0))]),
#         'LGBM': Pipeline([('LGBM', LGBMClassifier(verbosity=-1))]),
#         'XGB': Pipeline([('XGB', XGBClassifier())])
#     }
    
#     param_grids = {
#         'Logistic': {'Logistic__C': [0.1, 1, 10], 'Logistic__penalty': ['l1', 'l2']},
#         'SVC': {'SVC__C': [0.1, 1, 10], 'SVC__kernel': ['linear', 'rbf']},
#         'KNN': {'KNN__n_neighbors': [3, 5, 7], 'KNN__weights': ['uniform', 'distance']},
#         'MLP': {'MLP__hidden_layer_sizes': [(50,), (100,), (50, 50)], 'MLP__activation': ['relu', 'tanh'], 'MLP__alpha': [0.001, 0.01, 0.1]},
#         'RandomForest': {'RandomForest__n_estimators': [100, 200, 300], 'RandomForest__max_depth': [None, 5, 10]},
#         'ExtraTrees': {'ExtraTrees__n_estimators': [100, 200, 300], 'ExtraTrees__max_depth': [None, 5, 10]},
#         'CatBoost': {'CatBoost__iterations': [100, 200, 300], 'CatBoost__depth': [6, 8, 10]},
#         'LGBM': {'LGBM__n_estimators': [100, 200, 300], 'LGBM__max_depth': [None, 5, 10]},
#         'XGB': {'XGB__n_estimators': [100, 200, 300], 'XGB__max_depth': [None, 5, 10]}
#     }
    
#     grid_search_results = {}
#     for name, pipeline in pipelines.items():
#         with mlflow.start_run(nested=True):
#             mlflow.log_params(param_grids[name])
#             grid_search = GridSearchCV(pipeline, param_grids[name], cv=cv)
#             grid_search.fit(x_train, y_train)
#             grid_search_results[name] = grid_search
#             mlflow.log_metric(f"{name}_best_score", grid_search.best_score_)
#             mlflow.log_params(grid_search.best_params_)
    
#     return grid_search_results

def perform_grid_search(x_train, y_train):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    pipelines = {
        'Logistic': Pipeline([('MinMaxScale', MinMaxScaler()), ('Logistic', LogisticRegression(solver='liblinear'))]),
        'SVC': Pipeline([('MinMaxScale', MinMaxScaler()), ('SVC', SVC(probability=True))]),
        'KNN': Pipeline([('MinMaxScale', MinMaxScaler()), ('KNN', KNeighborsClassifier())]),
        'MLP': Pipeline([('MinMaxScale', MinMaxScaler()), ('MLP', MLPClassifier())]),
        'RandomForest': Pipeline([('RandomForest', RandomForestClassifier())]),
        'ExtraTrees': Pipeline([('ExtraTrees', ExtraTreesClassifier())]),
        'CatBoost': Pipeline([('CatBoost', CatBoostClassifier(verbose=0))]),
        'LGBM': Pipeline([('LGBM', LGBMClassifier(verbosity=-1))]),
        'XGB': Pipeline([('XGB', XGBClassifier())])
    }
    
    param_grids = {
        'Logistic': {'Logistic__C': [0.1, 1, 10], 'Logistic__penalty': ['l1', 'l2']},
        'SVC': {'SVC__C': [0.1, 1, 10], 'SVC__kernel': ['linear', 'rbf']},
        'KNN': {'KNN__n_neighbors': [3, 5, 7], 'KNN__weights': ['uniform', 'distance']},
        'MLP': {'MLP__hidden_layer_sizes': [(50,), (100,), (50, 50)], 'MLP__activation': ['relu', 'tanh'], 'MLP__alpha': [0.001, 0.01, 0.1]},
        'RandomForest': {'RandomForest__n_estimators': [100, 200, 300], 'RandomForest__max_depth': [None, 5, 10]},
        'ExtraTrees': {'ExtraTrees__n_estimators': [100, 200, 300], 'ExtraTrees__max_depth': [None, 5, 10]},
        'CatBoost': {'CatBoost__iterations': [100, 200, 300], 'CatBoost__depth': [6, 8, 10]},
        'LGBM': {'LGBM__n_estimators': [100, 200, 300], 'LGBM__max_depth': [None, 5, 10]},
        'XGB': {'XGB__n_estimators': [100, 200, 300], 'XGB__max_depth': [None, 5, 10]}
    }
    
    grid_search_results = {}
    for name, pipeline in pipelines.items():
        with mlflow.start_run(nested=True):
            grid_search = GridSearchCV(pipeline, param_grids[name], cv=cv)
            grid_search.fit(x_train, y_train)
            grid_search_results[name] = grid_search
            
            # Log parameters and metrics
            best_params = grid_search.best_params_
            mlflow.log_params(best_params)
            mlflow.log_metric(f"{name}_best_score", grid_search.best_score_)
    
    return grid_search_results


def print_best_scores(grid_search_results):
    print('5 FOLD BEST SCORE')
    print('--' * 15)
    for name, grid_search in grid_search_results.items():
        print(f'{name} best score: {round(grid_search.best_score_, 3)}')

def create_ensemble_models(grid_search_results):
    best_models = [grid_search.best_estimator_ for grid_search in grid_search_results.values()]
    
    ensemble_soft = VotingClassifier(
        estimators=[(name, model) for name, model in zip(grid_search_results.keys(), best_models)],
        voting='soft'
    )
    
    ensemble_hard = VotingClassifier(
        estimators=[(name, model) for name, model in zip(grid_search_results.keys(), best_models)],
        voting='hard'
    )
    
    return ensemble_soft, ensemble_hard

def evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    train_score = round(accuracy_score(y_train, train_pred), 3)
    test_score = round(accuracy_score(y_test, test_pred), 3)
    
    # Log detailed classification report
    report = classification_report(y_test, test_pred, output_dict=True)
    mlflow.log_metrics({f"{model.__class__.__name__}_precision": report['weighted avg']['precision'],
                        f"{model.__class__.__name__}_recall": report['weighted avg']['recall'],
                        f"{model.__class__.__name__}_f1-score": report['weighted avg']['f1-score']})
    
    # Create and log confusion matrix plot
    cm = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix - {model.__class__.__name__}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'{model.__class__.__name__}_confusion_matrix.png')
    mlflow.log_artifact(f'{model.__class__.__name__}_confusion_matrix.png')
    plt.close()
    
    return train_score, test_score

def create_stacking_models(grid_search_results):
    best_models = [grid_search.best_estimator_ for grid_search in grid_search_results.values()]
    
    stacking_logist = StackingClassifier(
        classifiers=best_models,
        meta_classifier=LogisticRegression()
    )
    
    stacking_lgbm = StackingClassifier(
        classifiers=best_models,
        meta_classifier=LGBMClassifier()
    )
    
    return stacking_logist, stacking_lgbm


def main(data_dir):
    mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
    mlflow.set_experiment("Heart Disease Classification")
    
    with mlflow.start_run():
        x_train, x_test, y_train, y_test = process_data(data_dir)
        mlflow.log_param("train_size", len(x_train))
        mlflow.log_param("test_size", len(x_test))
        
        grid_search_results = perform_grid_search(x_train, y_train)
        print_best_scores(grid_search_results)
        
        ensemble_soft, ensemble_hard = create_ensemble_models(grid_search_results)
        stacking_logist, stacking_lgbm = create_stacking_models(grid_search_results)
        
        models = [ensemble_soft, ensemble_hard, stacking_logist, stacking_lgbm]
        names = ['Ensemble Soft', 'Ensemble Hard', 'Stacking Logistic', 'Stacking LGBM']
        
        for name, model in zip(names, models):
            with mlflow.start_run(nested=True):
                mlflow.log_param("model_name", name)
                train_score, test_score = evaluate_model(model, x_train, y_train, x_test, y_test)
                mlflow.log_metric("train_score", train_score)
                mlflow.log_metric("test_score", test_score)
                print(f'{name} - Train Score: {train_score}, Test Score: {test_score}')
                
                # Log the model
                mlflow.sklearn.log_model(model, name)
        
        # Create and log a summary plot
        plt.figure(figsize=(12,8))
        scores = pd.DataFrame({
            'Model': names,
            'CV Score': [grid_search_results[name].best_score_ for name in names],
            'Test Score': [evaluate_model(models[names.index(name)], x_train, y_train, x_test, y_test)[1] for name in names]
        })
        scores = scores.melt(id_vars='Model', var_name='Type', value_name='Score')
        sns.barplot(x='Model', y='Score', hue='Type', data=scores)
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        mlflow.log_artifact('model_comparison.png')

# Usage
# X, y = load_your_data()  # Load your dataset
main("./data/dataset_heart.csv")
