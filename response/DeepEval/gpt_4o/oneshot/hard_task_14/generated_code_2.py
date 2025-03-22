import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path: Global Average Pooling + Fully connected layers to generate weights
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    dense_weights1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense_weights2 = Dense(units=3, activation='sigmoid')(dense_weights1)  # Assuming the input has 3 channels
    weights_reshaped = Reshape((1, 1, 3))(dense_weights2)  # Reshape to match input's channel dimension

    # Element-wise multiplication to scale the input feature map
    scaled_features = Multiply()([input_layer, weights_reshaped])

    # Branch path: 3x3 convolution to adjust channels
    conv_branch = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combining both paths by adding them
    combined = Add()([scaled_features, conv_branch])

    # Fully connected layers for final classification
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model