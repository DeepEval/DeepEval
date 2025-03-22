import keras
from keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1
    x1 = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    x1 = [layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(x) for x in x1]
    x1 = layers.Concatenate(axis=-1)(x1)

    # Block 2
    shape_before_reshape = keras.backend.int_shape(x1)
    x2 = layers.Reshape(target_shape=[shape_before_reshape[1], shape_before_reshape[2], 3, 32 // 3])(x1)
    x2 = layers.Permute([0, 1, 3, 2])(x2)
    x2 = layers.Reshape([-1, shape_before_reshape[1], shape_before_reshape[2], 32])(x2)

    # Block 3
    x3 = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', use_bias=False)(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)

    # Branch
    branch = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid')(inputs)

    # Combine
    merged = layers.Add()([branch, x3])
    merged = layers.BatchNormalization()(merged)
    merged = layers.Activation('relu')(merged)

    # Classification
    outputs = layers.Dense(units=10, activation='softmax')(merged)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Loss: {loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')