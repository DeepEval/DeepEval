import tensorflow as tf

def dl_model():

    model = tf.keras.models.Sequential()

    # Average pooling layer
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid'))

    # 1x1 convolutional layer
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='valid'))

    # Flatten the feature maps
    model.add(tf.keras.layers.Flatten())

    # First fully connected layer
    model.add(tf.keras.layers.Dense(units=1024, activation='relu'))

    # Dropout layer
    model.add(tf.keras.layers.Dropout(rate=0.2))

    # Second fully connected layer
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model