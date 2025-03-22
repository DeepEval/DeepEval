from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.layers import ReLU, SeparableConv2D

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # Main path
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = SeparableConv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Branch path
    branch_x = Conv2D(32, (1, 1), activation='relu')(x)
    
    # Concatenate the outputs of both paths
    y = Concatenate()([x, branch_x])
    y = Flatten()(y)
    
    # Fully connected layer
    outputs = Dense(10, activation='softmax')(y)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))