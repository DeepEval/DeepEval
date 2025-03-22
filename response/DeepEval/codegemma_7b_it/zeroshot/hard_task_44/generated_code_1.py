import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_img = keras.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_img)

    x_1 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x[0])
    x_2 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x[1])
    x_3 = layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(x[2])

    x = layers.Dropout(rate=0.2)(layers.concatenate([x_1, x_2, x_3]))

    # Block 2
    x = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x)

    x_1 = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x_3 = layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(x)
    x_4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    x_4 = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x_4)

    x = layers.concatenate([x_1, x_2, x_3, x_4])

    # Output layer
    x = layers.Flatten()(x)
    output = layers.Dense(units=10, activation='softmax')(x)

    # Model creation
    model = keras.Model(inputs=input_img, outputs=output)

    return model

# Compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])