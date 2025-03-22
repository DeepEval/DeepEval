from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define model
def dl_model():
    input_layer = layers.Input(shape=(32, 32, 3))

    # First block
    split_layer = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))([input_layer])
    conv1_1x1 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(split_layer[0])
    conv1_3x3 = layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu')(split_layer[1])
    conv1_5x5 = layers.SeparableConv2D(64, (5, 5), padding='same', activation='relu')(split_layer[2])
    concat_conv1 = layers.Concatenate(axis=-1)([conv1_1x1, conv1_3x3, conv1_5x5])

    # Second block
    branch_1x1 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(concat_conv1)
    branch_1x1_conv3x3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch_1x1)
    branch_3x3_conv1x1 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(concat_conv1)
    branch_3x3_conv3x3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch_3x3_conv1x1)
    branch_max_pool = layers.MaxPooling2D((2, 2))(concat_conv1)
    branch_max_pool_conv1x1 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(branch_max_pool)
    concat_conv2 = layers.Concatenate(axis=-1)([branch_1x1_conv3x3, branch_3x3_conv3x3, branch_max_pool_conv1x1])

    # Output layer
    flatten = layers.Flatten()(concat_conv2)
    output = layers.Dense(10, activation='softmax')(flatten)

    model = models.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=64)

    return model

# Train and evaluate model
model = dl_model()
model.evaluate(x_test, y_test)