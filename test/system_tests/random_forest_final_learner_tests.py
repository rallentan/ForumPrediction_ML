from unittest import TestCase

from application.config_loader import ConfigLoader
from application.metric_display import MetricDisplay
from domain.entities.rf_final_learner_input import RFFinalLearnerInput
from domain.learners.random_forest_final_learner import RandomForestFinalLearner
from external_services.spectrum_data_provider_adapter import SpectrumDataProviderAdapter


class RandomForestFinalLearnerTests(TestCase):

    def __init__(self, method_name='runTest'):
        super().__init__(method_name)
        self.learner_input = None

    def setUp(self):
        # Load configuration and settings from environment variables
        connection_info = ConfigLoader.load_database_connection_info()
        self.settings = ConfigLoader.load_settings()

        # Fetch invite data from the spectrum data provider
        print("Fetching raw data...")
        with SpectrumDataProviderAdapter(connection_info) as spectrum_data_provider:
            country_categories = spectrum_data_provider.get_countries()
            fluency_categories = spectrum_data_provider.get_fluencies()
            invites = spectrum_data_provider.get_invites()

        # Extract features from raw data
        print("Extracting features...")
        self.learner_input = RFFinalLearnerInput.from_invites(invites,
                                                              country_categories,
                                                              fluency_categories,
                                                              random_seed=self.settings['random_seed'])
        del invites
        MetricDisplay.print_data_statistics(self.learner_input.get_data_statistics())

        # Preprocess the data (i.e. encode features and targets to adjusted numerical values)
        print("Preprocessing data...")
        self.learner_input.preprocess(equalize_target_classes=False)

    def test_cross_validation(self):
        # Build and cross-validate model
        print("Cross-validating model...")
        learner = RandomForestFinalLearner(self.learner_input.get_feature_count(), self.settings['random_seed'])
        multi_scores = learner.cross_validate(self.learner_input,
                                                    ['roc_auc', 'f1', 'precision', 'recall', 'accuracy'])
        MetricDisplay.print_evaluation_metrics(multi_scores)

        # Inspect model internals
        print("Inspecting model internals...")
        inspections = learner.inspect(self.learner_input)
        MetricDisplay.print_feature_importances(inspections['importances'])
        MetricDisplay.print_feature_dependencies(inspections['dependencies'])
        MetricDisplay.plot_feature_importances(inspections['importances'])
        MetricDisplay.plot_feature_correlations(inspections['dependencies'])
        return
