import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, multiply, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels (RGB)
    
    # Global average pooling to compress the input features
    global_avg_pooling = GlobalAveragePooling2D()(input_layer)

    # Two fully connected layers to generate weights
    dense1 = Dense(units=128, activation='relu')(global_avg_pooling)
    dense2 = Dense(units=3, activation='sigmoid')(dense1)  # 3 channels in CIFAR-10 (RGB)

    # Reshape the weights to match the input shape (1, 1, 3)
    reshaped_weights = Reshape((1, 1, 3))(dense2)

    # Multiply element-wise with the input feature map
    scaled_features = multiply([input_layer, reshaped_weights])

    # Flatten the result and pass through another fully connected layer
    flatten_layer = Flatten()(scaled_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    model = Model(inputs=input_layer, outputs=output_layer)

    return model