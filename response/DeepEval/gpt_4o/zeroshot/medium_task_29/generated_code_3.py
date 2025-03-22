from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load and preprocess CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

    # Define the input
    input_img = Input(shape=(32, 32, 3))

    # First convolution layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    
    # First max pooling layer (1x1)
    pool_1x1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(x)
    flat_1x1 = Flatten()(pool_1x1)

    # Second max pooling layer (2x2)
    pool_2x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    flat_2x2 = Flatten()(pool_2x2)

    # Third max pooling layer (4x4)
    pool_4x4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(x)
    flat_4x4 = Flatten()(pool_4x4)

    # Concatenate pooled features
    concatenated_features = Concatenate()([flat_1x1, flat_2x2, flat_4x4])

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(concatenated_features)
    fc2 = Dense(64, activation='relu')(fc1)
    
    # Output layer
    output = Dense(10, activation='softmax')(fc2)

    # Define the model
    model = Model(inputs=input_img, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of how to use the model
model = dl_model()
model.summary()