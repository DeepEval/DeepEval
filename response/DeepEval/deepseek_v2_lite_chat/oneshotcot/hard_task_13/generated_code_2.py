import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Reshape

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the pixel values
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Define the model architecture
    def block(input_tensor, num_filters):
        conv1 = Conv2D(filters=num_filters, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=num_filters, kernel_size=(5, 5), padding='same', activation='relu')(conv1)
        maxpool = MaxPooling2D(pool_size=(3, 3), padding='same')(conv2)
        return maxpool

    # First block
    conv1 = block(input_tensor=x_train, num_filters=32)
    conv2 = block(input_tensor=conv1, num_filters=64)
    conv3 = block(input_tensor=conv2, num_filters=64)
    conv4 = block(input_tensor=conv3, num_filters=64)
    concat = keras.layers.concatenate([conv1, conv2, conv3, conv4])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat)
    flat = Flatten()(batch_norm)

    # Second block
    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense1)
    avg_pool = GlobalAveragePooling2D()(dense2)

    # Fully connected layers
    reshape = Reshape((2048,))(avg_pool)
    dense3 = Dense(512, activation='relu')(reshape)
    dense4 = Dense(10, activation='softmax')(dense3)

    # Construct the model
    model = Model(inputs=input_tensor, outputs=dense4)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()