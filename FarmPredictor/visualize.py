import matplotlib.pyplot as plt

def plot_results(model, X_test, y_test):
    # Reshape input data for LSTM
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    predictions = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, color='blue', label='Actual')
    plt.plot(predictions, color='red', label='Predicted')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Time')
    plt.ylabel('Agricultural Variable')
    plt.legend()
    plt.show()
