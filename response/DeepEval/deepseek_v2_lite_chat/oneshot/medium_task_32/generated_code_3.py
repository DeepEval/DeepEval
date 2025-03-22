import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))  # Assuming the images are 32x32 pixels

    # Split the input into three groups along the last dimension
    group1 = Lambda(lambda x: keras.layers.Lambda(lambda inp: tf.split(inp, [16, 16, 8], axis=-1))(x))(input_layer)
    group2 = Lambda(lambda x: keras.layers.Lambda(lambda inp: tf.split(inp, [16, 16, 8], axis=-1))(x))(input_layer)
    group3 = Lambda(lambda x: keras.layers.Lambda(lambda inp: tf.split(inp, [16, 16, 8], axis=-1))(x))(input_layer)

    # Depthwise separable convolutional layers for each group
    def depthwise_separable_layer(input_tensor, filters, kernel_size):
        conv = Conv2D(filters, kernel_size, padding='same', activation='relu')(input_tensor)
        conv = Conv2D(filters, (1, 1), padding='same', activation='relu')(conv)
        return conv

    # 1x1 convolutional layer
    conv1x1 = depthwise_separable_layer(group1, filters=64, kernel_size=1)
    # 3x3 convolutional layer
    conv3x3 = depthwise_separable_layer(group2, filters=64, kernel_size=3)
    # 5x5 convolutional layer
    conv5x5 = depthwise_separable_layer(group3, filters=64, kernel_size=5)

    # Concatenate the outputs of the three paths
    fused_features = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5])

    # Batch normalization and flattening
    bn = BatchNormalization()(fused_features)
    flat = Flatten()(bn)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])