import keras
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense, AveragePooling2D
from keras.layers.merge import concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        # Splitting the input into three groups
        groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        # First group remains unchanged
        unchanged_group = groups[0]
        # Second group goes through a 3x3 convolutional layer
        conv2d_group = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(groups[1])
        # Third group goes through an additional 3x3 convolution
        additional_conv_group = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(groups[2])
        # Concatenate the outputs of the three groups
        concatenated = concatenate([unchanged_group, conv2d_group, additional_conv_group])
        # Additional layers for the main path can be added here
        # Example: Flattening and additional dense layers
        flattened = Flatten()(concatenated)
        dense1 = Dense(units=128, activation='relu')(flattened)
        dense2 = Dense(units=10, activation='softmax')(dense1)
        # Model for the main path
        main_model = Model(inputs=input_layer, outputs=dense2)
        
        return main_model
    
    # Branch path
    def branch_path(input_tensor):
        # 1x1 convolutional layer
        conv1x1 = Conv2D(filters=64, kernel_size=(1, 1))(input_tensor)
        # Additional layers for the branch path can be added here
        # Example: AveragePooling2D for downsampling
        avg_pool = AveragePooling2D(pool_size=(2, 2))(conv1x1)
        # Flatten and fully connected layers for the branch path
        flattened = Flatten()(avg_pool)
        dense1 = Dense(units=128, activation='relu')(flattened)
        dense2 = Dense(units=10, activation='softmax')(dense1)
        # Model for the branch path
        branch_model = Model(inputs=input_tensor, outputs=dense2)
        
        return branch_model
    
    # Combine the models for the main and branch paths
    main_model = main_path(input_layer)
    branch_model = branch_path(input_layer)
    combined_model = Add()([main_model.output, branch_model.output])
    combined_model = Flatten()(combined_model)
    output_layer = Dense(units=10, activation='softmax')(combined_model)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])