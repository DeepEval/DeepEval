import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, SeparableConv2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(main_path[0])
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(main_path[1])
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(main_path[2])
    main_output = Concatenate(axis=3)([conv1x1, conv3x3, conv5x5])
    
    # Branch path
    branch_output = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Fuse both paths
    fused_output = tf.add(main_output, branch_output)
    
    # Flatten the output
    flatten_layer = Flatten()(fused_output)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()