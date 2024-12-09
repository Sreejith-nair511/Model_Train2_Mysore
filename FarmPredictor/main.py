from preprocess import load_and_preprocess_data
from model import build_and_train_model
from evaluate import evaluate_model
from visualize import plot_results

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data('data/rainfall.csv')

# Build and train model
model = build_and_train_model(X_train, y_train)

# Evaluate model
evaluate_model(model, X_test, y_test)

# Visualize results
plot_results(model, X_test, y_test)
