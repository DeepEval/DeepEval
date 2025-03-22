import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Reshape, Multiply, Add, Dense, BatchNormalization, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        fc1 = Dense(units=32, activation='relu')(avg_pool)
        fc2 = Dense(units=32, activation='relu')(fc1)
        reshape = Reshape(target_shape=(1, 1, 32))(fc2)
        weighted_features = Multiply()([input_tensor, reshape])
        return weighted_features

    block1_output = block1(input_layer)

    # Block 2
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(block1_output)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Branch from Block 1
    branch_from_block1 = block1(max_pool)

    # Fusion of main path and branch
    fused_output = Add()([max_pool, branch_from_block1])

    # Flatten and classify
    flatten = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage:
# model = dl_model()
# model.summary()