import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate, Conv2D, Add, BatchNormalization, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_output = GlobalAveragePooling2D()(input_layer)
    main_dense1 = Dense(units=128, activation='relu')(main_output)
    main_dense2 = Dense(units=64, activation='relu')(main_dense1)
    
    # Create weights from the main path
    weight1 = Dense(units=main_dense2.shape[1] * main_output.shape[1] * main_output.shape[2] * 3)(main_dense2)
    weight1 = tf.reshape(weight1, (-1, 3, main_output.shape[1], main_output.shape[2]))  # Reshape for multiplication
    weight1 = tf.transpose(weight1, (0, 2, 3, 1))  # Transpose for element-wise multiplication
    
    # Main path weights applied to original feature map
    main_feature_map = tf.nn.conv2d(input_layer, weight1, strides=[1, 1, 1, 1], padding='same')
    main_feature_map = tf.reshape(main_feature_map, (-1, 3, input_layer.shape[1], input_layer.shape[2]))
    
    # Branch path
    branch_output = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_output = BatchNormalization()(branch_output)
    branch_output = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_output)
    branch_output = BatchNormalization()(branch_output)
    branch_output = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_output)
    branch_output = BatchNormalization()(branch_output)
    
    # Add both paths
    combined_output = Add()([main_feature_map, branch_output])
    
    # Fully connected layers
    combined_output = Flatten()(combined_output)
    combined_output = Dense(units=128, activation='relu')(combined_output)
    combined_output = Dense(units=64, activation='relu')(combined_output)
    output_layer = Dense(units=10, activation='softmax')(combined_output)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])