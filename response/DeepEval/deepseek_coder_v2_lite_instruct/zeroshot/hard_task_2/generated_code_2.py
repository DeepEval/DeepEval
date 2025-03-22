import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Process each group through a series of convolutions
    conv_groups = []
    for group in split_layer:
        # 1x1 convolution
        conv1x1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(group)
        # 3x3 convolution
        conv3x3 = Conv2D(32, kernel_size=(3, 3), activation='relu')(conv1x1)
        # Another 1x1 convolution
        conv1x1_2 = Conv2D(32, kernel_size=(1, 1), activation='relu')(conv3x3)
        conv_groups.append(conv1x1_2)
    
    # Combine the outputs from the three groups using addition
    combined_features = Add()(conv_groups)
    
    # Add the combined features back to the original input
    main_path = Add()([input_layer, combined_features])
    
    # Flatten the combined features
    flattened_features = Flatten()(main_path)
    
    # Fully connected layer for classification
    output_layer = Dense(10, activation='softmax')(flattened_features)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage
model = dl_model()
model.summary()