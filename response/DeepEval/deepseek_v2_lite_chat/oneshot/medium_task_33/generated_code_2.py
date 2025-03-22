import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_shape = (32, 32, 3)  # Input shape is (32, 32, 3) since each pixel is in RGB format

    # Split the input into three channel groups
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(Input(shape=input_shape))

    # Apply separable convolutions
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(split[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(split[1])
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(split[2])

    # MaxPooling
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Concatenate the outputs
    concat = Concatenate()([pool1, pool2, pool3])

    # Flatten and pass through fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=Input(shape=input_shape), outputs=output)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Optional: Print model summary
model.summary()