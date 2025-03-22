from keras.layers import Input, AveragePooling2D, Flatten, Dense, Reshape, Concatenate, Conv2D, GlobalAveragePooling2D
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical

def dl_model():
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = x_train[..., None]  # Add channel dimension
    x_test = x_test[..., None]
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    input_layer = Input(shape=(28, 28, 1))

    # Block 1 - Parallel Average Pooling
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    concat1 = Concatenate()([flat1, flat2, flat3])

    # Fully connected layer after Block 1
    fc1 = Dense(128, activation='relu')(concat1)

    # Reshape for Block 2
    reshaped = Reshape((4, 4, 8))(fc1)  # Assumed shape for demonstration

    # Block 2 - Feature Extraction
    # Branch 1 - 1x1 conv followed by 3x3 conv
    conv1x1_1 = Conv2D(16, (1, 1), activation='relu', padding='same')(reshaped)
    conv3x3_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1x1_1)

    # Branch 2 - 1x1 conv, 1x7 conv, 7x1 conv, followed by 3x3 conv
    conv1x1_2 = Conv2D(16, (1, 1), activation='relu', padding='same')(reshaped)
    conv1x7 = Conv2D(16, (1, 7), activation='relu', padding='same')(conv1x1_2)
    conv7x1 = Conv2D(16, (7, 1), activation='relu', padding='same')(conv1x7)
    conv3x3_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7x1)

    # Branch 3 - Average Pooling
    pool_b3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped)

    # Concatenate outputs from all branches in Block 2
    concat2 = Concatenate()([conv3x3_1, conv3x3_2, pool_b3])

    # Global average pooling before final Dense layers
    gap = GlobalAveragePooling2D()(concat2)

    # Fully connected layers for classification
    fc2 = Dense(256, activation='relu')(gap)
    output = Dense(10, activation='softmax')(fc2)

    # Construct model
    model = Model(inputs=input_layer, outputs=output)

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of creating the model
model = dl_model()
print(model.summary())