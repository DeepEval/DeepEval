import keras
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the channel
    group1, group2, group3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    
    # Separable convolutional layers for multi-scale feature extraction
    conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu')(group1)
    conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu')(group2)
    conv3 = SeparableConv2D(filters=64, kernel_size=(5, 5), activation='relu')(group3)
    
    # Concatenate the outputs from the three groups
    concatenated = Concatenate()(
        [conv1, conv2, conv3]
    )
    
    # 1x1 convolutional layer for aligning channel dimensions
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concatenated)
    
    # Batch normalization and max pooling for the main path
    main_path_output = MaxPooling2D(pool_size=(4, 4))(concatenated)
    batch_norm_main = BatchNormalization()(main_path_output)
    flatten_main = Flatten()(batch_norm_main)
    
    # Fully connected layers for the main path
    dense1_main = Dense(units=128, activation='relu')(flatten_main)
    dense2_main = Dense(units=64, activation='relu')(dense1_main)
    
    # Softmax activation for the main path's output
    output_main = Dense(units=10, activation='softmax')(dense2_main)
    
    # Add the branch path
    input_branch = Input(shape=(32, 32, 3))
    branch_output = SeparableConv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_branch)
    batch_norm_branch = BatchNormalization()(branch_output)
    flatten_branch = Flatten()(batch_norm_branch)
    dense1_branch = Dense(units=128, activation='relu')(flatten_branch)
    dense2_branch = Dense(units=64, activation='relu')(dense1_branch)
    
    # Fusion of the main path and the branch path
    fused_output = keras.layers.Add()([output_main, dense2_branch])
    
    # Final flattening and fully connected layers
    final_flatten = Flatten()(fused_output)
    dense3 = Dense(units=128, activation='relu')(final_flatten)
    output_layer = Dense(units=10, activation='softmax')(dense3)
    
    # Model construction
    model = Model(inputs=[input_layer, input_branch], outputs=[output_main, output_layer])
    
    return model

# Instantiate the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])