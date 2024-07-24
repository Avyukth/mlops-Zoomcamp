import os
import pandas as pd
from typing import Tuple
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset, DataQualityTestPreset
from evidently.tests import TestColumnDrift
import mlflow

class EvidentlyReporter:
    def __init__(self, output_dir: str):
        """
        Initialize the EvidentlyReporter.

        Args:
            output_dir (str): Directory to save the Evidently reports.
        """
        self.output_dir = output_dir

    def generate_reports(self, x_train: pd.DataFrame, y_train: pd.Series, 
                         x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[str, str]:
        """
        Generate Evidently Test Suite and Report.

        Args:
            x_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            x_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target.

        Returns:
            Tuple[str, str]: Paths to the generated report and test suite.
        """
        # Combine features and target for each dataset
        train_data = pd.concat([x_train, y_train], axis=1)
        test_data = pd.concat([x_test, y_test], axis=1)

        # Create an Evidently Report
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset()
        ])

        report.run(reference_data=train_data, current_data=test_data)
        report_path = os.path.join(self.output_dir, "evidently_report.html")
        report.save_html(report_path)
        mlflow.log_artifact(report_path, "evidently_reports")

        # Create an Evidently Test Suite
        test_suite = TestSuite(tests=[
            DataStabilityTestPreset(),
            DataQualityTestPreset(),
            TestColumnDrift(column_name="age"),
            TestColumnDrift(column_name="sex"),
            TestColumnDrift(column_name="chest_pain_type"),
            TestColumnDrift(column_name="resting_blood_pressure"),
        ])

        test_suite.run(reference_data=train_data, current_data=test_data)
        test_suite_path = os.path.join(self.output_dir, "evidently_test_suite.html")
        test_suite.save_html(test_suite_path)
        mlflow.log_artifact(test_suite_path, "evidently_reports")

        print(f"Evidently reports saved to {report_path} and {test_suite_path}")
        return report_path, test_suite_path

    def log_reports_to_mlflow(self, report_path: str, test_suite_path: str):
        """
        Log the generated reports to MLflow.

        Args:
            report_path (str): Path to the Evidently report.
            test_suite_path (str): Path to the Evidently test suite.
        """
        mlflow.log_artifact(report_path, "evidently_reports")
        mlflow.log_artifact(test_suite_path, "evidently_reports")
