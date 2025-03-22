from tensorflow.keras.layers import Input, Conv2D, Add, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First path with three sequential convolutional layers
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    
    # Second path with a separate convolutional layer processing the input directly
    direct_conv = Conv2D(128, (3, 3), activation='relu', padding='same')(input_layer)
    
    # Adding the outputs from the first path's third layer and the direct path
    added = Add()([conv3, direct_conv])
    
    # Flatten and pass through fully connected layers for classification
    flatten = Flatten()(added)
    fc1 = Dense(256, activation='relu')(flatten)
    output_layer = Dense(10, activation='softmax')(fc1)  # CIFAR-10 has 10 classes
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage
# model = dl_model()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load CIFAR-10 data
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))