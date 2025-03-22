import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Process each group
    processed_groups = []
    for group in split_layer:
        group = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group)
        group = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(group)
        group = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group)
        processed_groups.append(group)
    
    # Combine the processed groups using addition
    combined_path = Add()(processed_groups)
    
    # Add the combined path back to the original input
    main_path = Add()([combined_path, input_layer])
    
    # Flatten the combined features
    flatten_layer = Flatten()(main_path)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model