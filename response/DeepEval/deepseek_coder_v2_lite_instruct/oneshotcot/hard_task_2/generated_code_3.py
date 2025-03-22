import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Process each group through a series of convolutions
    paths = []
    for group in split:
        path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group)
        path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path)
        path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path)
        paths.append(path)
    
    # Combine the outputs from the three groups using addition
    main_path = Add()(paths)
    
    # Add the main path back to the original input
    combined = Add()([main_path, input_layer])
    
    # Flatten the combined features into a one-dimensional vector
    flatten_layer = Flatten()(combined)
    
    # Feed the flattened features into a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()