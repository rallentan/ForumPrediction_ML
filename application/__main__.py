# Notes:
# - By convention, 'X' denotes a sample set of input data, while 'y' denotes the corresponding set of target
# classifications the domain must predict.
import csv

from application.config_loader import ConfigLoader
from application.metric_display import MetricDisplay
from domain.entities.rf_final_learner_input import RFFinalLearnerInput
from domain.learners.random_forest_final_learner import RandomForestFinalLearner
from external_services.spectrum_data_provider_adapter import SpectrumDataProviderAdapter


def main():
    # Load configuration and settings from environment variables
    connection_info = ConfigLoader.load_database_connection_info()
    settings = ConfigLoader.load_settings()

    # Fetch invite data from the spectrum data provider
    print("Fetching raw data...")
    with SpectrumDataProviderAdapter(connection_info) as spectrum_data_provider:
        country_categories = spectrum_data_provider.get_countries()
        fluency_categories = spectrum_data_provider.get_fluencies()
        invites = spectrum_data_provider.get_invites()

    # Extract features from raw data
    print("Extracting features...")
    final_learner_input = RFFinalLearnerInput.from_invites(invites,
                                                           country_categories,
                                                           fluency_categories,
                                                           random_seed=settings['random_seed'])
    del invites
    MetricDisplay.print_data_statistics(final_learner_input.get_data_statistics())
    final_learner_input.save_as_csv("extracted_features.csv")

    # Preprocess the data (i.e. encode features and targets to adjusted numerical values)
    print("Preprocessing data...")
    final_learner_input.preprocess(equalize_target_classes=True)
    final_learner_input.save_as_csv("preprocessed_features.csv")

    # Build and cross-validate model
    print("Cross-validating model...")
    final_learner = RandomForestFinalLearner(final_learner_input.get_feature_count(), settings['random_seed'])
    multi_scores = final_learner.cross_validate(final_learner_input,
                                                ['roc_auc', 'f1', 'precision', 'recall', 'accuracy'])
    MetricDisplay.print_evaluation_metrics(multi_scores)

    # Inspect model internals
    print("Inspecting model internals...")
    inspections = final_learner.inspect(final_learner_input)
    MetricDisplay.print_feature_importances(inspections['importances'])
    MetricDisplay.print_feature_dependencies(inspections['dependencies'])
    MetricDisplay.plot_feature_importances(inspections['importances'])
    MetricDisplay.plot_feature_correlations(inspections['dependencies'])
    return


if __name__ == "__main__":
    main()
