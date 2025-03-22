import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Flatten, Concatenate, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    conv1_bn = BatchNormalization()(conv1)
    conv1_relu = keras.activations.relu(conv1_bn)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1_relu)
    conv2_bn = BatchNormalization()(conv2)
    conv2_relu = keras.activations.relu(conv2_bn)

    # Branch path
    branch_input = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    branch_input_bn = BatchNormalization()(branch_input)
    branch_relu = keras.activations.relu(branch_input_bn)

    # Combine paths
    combined = Add()[(1, 1)]([conv2_relu, branch_relu])

    # Block 1: Main path and branch path combined
    block1_output = combined

    # Second block
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(block1_output)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1_output)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(block1_output)

    # Flatten and concatenate
    flat1 = Flatten()(pool3)
    concat = Concatenate()([flat1, pool2, pool1])

    # Dense layers
    dense1 = Dense(units=128, activation='relu')(concat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.summary()