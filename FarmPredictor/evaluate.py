import tensorflow as tf

def evaluate_model(model, X_test, y_test):
    # Reshape input data for LSTM
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    predictions = model.predict(X_test)
    mse = tf.keras.losses.mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
