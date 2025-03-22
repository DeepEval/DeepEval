import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First branch with 3x3 convolution
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second branch with 5x5 convolution
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine features from both branches using addition
    combined_features = Add()([conv1, conv2])

    # Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(combined_features)

    # Fully connected layers to generate attention weights
    dense1 = Dense(units=64, activation='relu')(global_avg_pool)
    attention_weights = Dense(units=2, activation='softmax')(dense1)

    # Split attention weights
    weight1, weight2 = attention_weights[..., 0:1], attention_weights[..., 1:2]

    # Multiply each branch's output by corresponding attention weight
    weighted_output1 = Multiply()([conv1, weight1])
    weighted_output2 = Multiply()([conv2, weight2])

    # Add the weighted outputs
    final_weighted_output = Add()([weighted_output1, weighted_output2])

    # Flatten and final fully connected layer for classification
    flatten = Flatten()(final_weighted_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model