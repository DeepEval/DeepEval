import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Multiply

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First branch with 3x3 convolution
    conv_branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second branch with 5x5 convolution
    conv_branch2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Adding the two branches
    combined_branches = Add()([conv_branch1, conv_branch2])

    # Global Average Pooling layer
    global_avg_pool = GlobalAveragePooling2D()(combined_branches)

    # Fully connected layers for attention weights
    attention_dense1 = Dense(units=64, activation='relu')(global_avg_pool)
    attention_weights = Dense(units=2, activation='softmax')(attention_dense1)  # Two weights for two branches

    # Splitting attention weights for two branches
    weight_branch1 = attention_weights[:, 0]
    weight_branch2 = attention_weights[:, 1]

    # Expanding dims to match the shape of branch outputs
    weight_branch1 = keras.layers.Lambda(lambda x: keras.backend.expand_dims(x, axis=-1))(weight_branch1)
    weight_branch2 = keras.layers.Lambda(lambda x: keras.backend.expand_dims(x, axis=-1))(weight_branch2)

    # Applying weights to each branch
    weighted_output1 = Multiply()([conv_branch1, weight_branch1])
    weighted_output2 = Multiply()([conv_branch2, weight_branch2])

    # Adding weighted branches
    final_weighted_output = Add()([weighted_output1, weighted_output2])

    # Final fully connected layer to produce classification output
    final_output = Dense(units=10, activation='softmax')(final_weighted_output)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=final_output)

    return model