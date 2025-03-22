import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, Dense, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional branch 1 with 3x3 kernel
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    # Convolutional branch 2 with 5x5 kernel
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(conv1)

    # Pooling layer for branch 1
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    # Pooling layer for branch 2
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Add branch 1 and branch 2
    add_layer = Add()([pool1, pool2])

    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(add_layer)

    # Fully connected layer for attention weights
    fc1 = Dense(units=512, activation='relu')(avg_pool)
    fc2 = Dense(units=256, activation='relu')(fc1)

    # Softmax function for attention weights
    attention_weights = Dense(units=2, activation='softmax', name='attention_weights')(fc1)

    # Multiply branch outputs with their corresponding attention weights
    weighted_output = keras.layers.Multiply()([add_layer, attention_weights] * 2)

    # Final dense layer for classification
    output_layer = Dense(units=10, activation='softmax')(weighted_output)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and print the model
model = dl_model()
model.summary()