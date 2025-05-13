import numpy as np

def inverse_transform_predictions(scaled_predictions, scaler, target_feature_index, num_total_features):
    scaled_predictions_flat = scaled_predictions.reshape(-1, 1)
    dummy_array = np.zeros((len(scaled_predictions_flat), num_total_features))
    dummy_array[:, target_feature_index] = scaled_predictions_flat[:, 0]
    inversed_dummy = scaler.inverse_transform(dummy_array)
    inversed_target_flat = inversed_dummy[:, target_feature_index]
    inversed_target = inversed_target_flat.reshape(scaled_predictions.shape)
    return inversed_target