import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    # Split the input into three groups
    splits = Lambda(lambda tensor: tf.split(tensor, num_or_size_splits=3, axis=-1))(input_layer)
    
    # First group remains unchanged
    main_path_first_group = splits[0]
    
    # Second group undergoes feature extraction via a 3x3 convolutional layer
    main_path_second_group = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
    
    # Combine the second and third groups before passing through an additional 3x3 convolution
    combined_second_third_groups = Concatenate()([main_path_second_group, splits[2]])
    main_path_third_group = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(combined_second_third_groups)
    
    # Concatenate the outputs of all three groups
    main_path_output = Concatenate()([main_path_first_group, main_path_second_group, main_path_third_group])
    
    # Branch Path
    # Process the input with a 1x1 convolutional layer
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse the outputs from both paths through addition
    fused_output = tf.add(main_path_output, branch_path)
    
    # Flatten the combined output
    flattened_output = Flatten()(fused_output)
    
    # Pass the flattened output through a fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flattened_output)
    
    # Construct and return the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()