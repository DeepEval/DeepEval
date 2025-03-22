import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Add, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 3x3 convolutions
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1_1)
    branch1_output = conv1_2

    # Branch 2: 5x5 convolutions
    conv2_1 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)
    conv2_2 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(conv2_1)
    branch2_output = conv2_2

    # Combine branches
    merged_features = Add()([branch1_output, branch2_output])

    # Global average pooling
    pooled_features = GlobalAveragePooling2D()(merged_features)

    # Attention weights
    attention_layer1 = Dense(units=10, activation='softmax')(pooled_features)
    attention_layer2 = Dense(units=10, activation='softmax')(pooled_features)

    # Weighted sum
    weighted_output1 = attention_layer1 * branch1_output
    weighted_output2 = attention_layer2 * branch2_output

    final_output = Add()([weighted_output1, weighted_output2])

    # Final classification layer
    output_layer = Dense(units=10, activation='softmax')(final_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model