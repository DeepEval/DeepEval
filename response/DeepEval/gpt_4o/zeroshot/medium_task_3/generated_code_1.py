from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load MNIST dataset to understand input shape
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    input_shape = x_train.shape[1:]  # (28, 28, 1)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # First convolutional block
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x1 = MaxPooling2D((2, 2))(x1)
    
    # Second convolutional block
    x2 = Conv2D(1, (3, 3), activation='relu', padding='same')(x1)
    x2 = MaxPooling2D((2, 2))(x2)
    
    # Adding input to the last convolution output to maintain the same channel dimension
    x2_resized = Conv2D(1, (1, 1), activation='relu', padding='same')(inputs)
    added = Add()([x2, x2_resized])
    
    # Flatten and add a dense layer for classification
    x_flatten = Flatten()(added)
    outputs = Dense(10, activation='softmax')(x_flatten)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Instantiate and summarize the model
model = dl_model()
model.summary()