from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first parallel branch
    branch1 = Conv2D(32, (1, 1), activation='relu')(input_shape)
    branch1 = Conv2D(32, (1, 3), activation='relu')(branch1)
    branch1 = Conv2D(32, (3, 1), activation='relu')(branch1)
    branch1 = Flatten()(branch1)

    # Define the second parallel branch
    branch2 = Conv2D(32, (1, 1), activation='relu')(input_shape)
    branch2 = Conv2D(32, (1, 3), activation='relu')(branch2)
    branch2 = Conv2D(32, (3, 1), activation='relu')(branch2)
    branch2 = Flatten()(branch2)

    # Concatenate the two parallel branches
    x = Concatenate()([branch1, branch2])

    # Define the main pathway
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_shape, outputs=x)

    # Return the constructed model
    return model


# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss:.2f}, Accuracy: {accuracy:.2f}')