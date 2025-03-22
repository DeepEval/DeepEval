import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from keras.models import Model

def block(input_tensor):
    # Batch normalization and ReLU activation
    batch_norm = BatchNormalization()(input_tensor)
    relu_activation = ReLU()(batch_norm)
    # 3x3 convolutional layer
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu_activation)
    # Concatenate original input with new features
    output_tensor = Concatenate()([input_tensor, conv_layer])
    return output_tensor

def pathway(input_tensor):
    # Repeated block structure executed three times
    for _ in range(3):
        input_tensor = block(input_tensor)
    return input_tensor

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Create two pathways
    pathway1_output = pathway(input_layer)
    pathway2_output = pathway(input_layer)

    # Merge both pathways
    merged_output = Concatenate()([pathway1_output, pathway2_output])

    # Flatten the merged output
    flatten_layer = Flatten()(merged_output)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate the model
model = dl_model()