import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Add
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    main_path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    
    # Split the input into three groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    
    # Process each group with different convolutional layers
    processed_groups = [main_path1, main_path2, main_path3]
    
    # Concatenate the outputs from the three groups
    concatenated_main = Concatenate(axis=3)(processed_groups)
    
    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the branch path to the main path
    fused_features = Add()([concatenated_main, branch_path])
    
    # Flatten the fused features
    flatten_layer = Flatten()(fused_features)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model