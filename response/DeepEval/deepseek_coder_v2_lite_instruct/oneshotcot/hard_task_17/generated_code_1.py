import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Multiply, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        gap = GlobalAveragePooling2D()(input_tensor)
        fc1 = Dense(units=32, activation='relu')(gap)
        fc2 = Dense(units=32, activation='relu')(fc1)
        reshaped_weights = fc2[:, None, None, :]
        reshaped_weights = reshaped_weights.reshape(input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], -1)
        weighted_features = Multiply()([input_tensor, reshaped_weights])
        return weighted_features
    
    block1_output = block1(input_layer)

    # Block 2
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(block1_output)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
    max_pooling = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Branch from Block 1
    branch = block1(max_pooling)

    # Fuse the main path and the branch through addition
    fused_output = Add()([max_pooling, branch])

    # Flatten the output
    flattened = Flatten()(fused_output)

    # Output layer
    dense1 = Dense(units=64, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Build the model
model = dl_model()
model.summary()