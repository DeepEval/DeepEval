import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Lambda, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Lambda, BatchNormalization, Flatten, Dense

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the RGB codes to the range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # Model architecture
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Branch path
    branch_input = input_layer
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_input)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_input)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch_input)

    # Combine paths
    combined = Add()([conv2, branch3])

    # Pooling layer
    pool = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(combined)

    # Block 1: Batch normalization and flattening
    batch_norm1 = BatchNormalization()(pool)
    flat1 = Flatten()(batch_norm1)

    # Block 2: Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flat1)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate the model
model = dl_model()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test))

# Evaluate the model on the test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])