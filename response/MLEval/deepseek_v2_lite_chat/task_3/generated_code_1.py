import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate a simple sequence that follows a linear pattern
def generate_sequence(start, end, num_terms):
    sequence = []
    current = start
    for _ in range(num_terms):
        sequence.append(current)
        current += 1
    return sequence

# Prepare the sequence
def prepare_sequence():
    sequence = generate_sequence(0, 10, 100)  # Example sequence starting from 0 and increasing by 1
    # X will be the sequence itself (time steps), Y will be the corresponding sequence values
    X, Y = np.array(sequence[:-1]).reshape(-1, 1), np.array(sequence[1:])
    return train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the linear regression model
def train_model(X_train, Y_train):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

# Predict the next term in the sequence
def predict_next_term(model, X_test):
    return model.predict(X_test)

# Main function to execute the process
def method():
    X_train, X_test, Y_train, Y_test = prepare_sequence()
    model = train_model(X_train, Y_train)
    predicted_next_term = predict_next_term(model, X_test)
    return predicted_next_term

# Call the method for validation
if __name__ == "__main__":
    output = method()
    print("Predicted next term:", output)

    # Plot the original sequence and the predicted sequence
    plt.figure(figsize=(10, 6))
    plt.plot(generate_sequence(0, 10, 15), label='Original sequence')
    plt.plot(output, label='Predicted sequence', linestyle='dashed')
    plt.title('Predicted Next Term in the Sequence')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()