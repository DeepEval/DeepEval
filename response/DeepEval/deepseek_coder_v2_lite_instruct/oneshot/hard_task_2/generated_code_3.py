import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Define the convolutions for each group
    def conv_group(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv2)
        return conv3
    
    # Apply convolutions to each group
    conv1 = conv_group(split_layer[0])
    conv2 = conv_group(split_layer[1])
    conv3 = conv_group(split_layer[2])
    
    # Combine the outputs from the three groups using addition
    combined = Add()([conv1, conv2, conv3])
    
    # Add the combined features back to the original input
    added = Add()([combined, input_layer])
    
    # Flatten the combined features
    flatten_layer = Flatten()(added)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and compile the model (optional, depending on requirements)
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])