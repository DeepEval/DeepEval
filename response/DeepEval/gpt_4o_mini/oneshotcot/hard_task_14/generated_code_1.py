import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, Add, Flatten
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Main path: Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(input_layer)

    # Fully connected layers to generate weights
    dense1 = Dense(units=512, activation='relu')(global_avg_pool)
    dense2 = Dense(units=3 * 1 * 1, activation='sigmoid')(dense1)  # Output size matches channels of input (3)

    # Reshape the weights to match the input layer's shape
    reshaped_weights = Reshape((1, 1, 3))(dense2)  # Reshape to (1, 1, 3)

    # Multiply element-wise with the original feature map
    weighted_feature_map = Multiply()([input_layer, reshaped_weights])

    # Branch path: Convolution to match the input channels
    branch_path = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Combine the outputs from both paths
    combined_output = Add()([weighted_feature_map, branch_path])

    # Flatten the combined output for the fully connected layers
    flatten_layer = Flatten()(combined_output)

    # Fully connected layers for classification
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    dense4 = Dense(units=64, activation='relu')(dense3)
    output_layer = Dense(units=10, activation='softmax')(dense4)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model