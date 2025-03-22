import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, Lambda, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Main path
    main_path_group1 = split_layer[0]  # Remains unchanged
    main_path_group2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split_layer[1])
    main_path_group3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split_layer[2])
    combined_main_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(Add()([main_path_group2, main_path_group3]))
    
    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Fusion of main and branch paths
    fused_output = Add()([combined_main_path, branch_path])
    
    # Flatten the fused output
    flatten_layer = Flatten()(fused_output)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model