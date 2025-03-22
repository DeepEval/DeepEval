import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    group1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Convolutional layers for each group
    def conv_layer(group):
        conv1 = Conv2D(64, (1, 1), activation='relu')(group[0])
        conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
        conv3 = Conv2D(64, (1, 1), activation='relu')(conv2)
        return conv3
    
    # Apply the convolutional layers to each group
    conv_outs = [conv_layer(group) for group in group1]
    
    # Concatenate the output of the three groups
    main_path = Add()([conv_outs[0], conv_outs[1], conv_outs[2]])
    
    # Concatenate the main path with the original input
    combined = Add()([input_layer, main_path])
    
    # Flatten the combined features
    flatten = Flatten()(combined)
    
    # Fully connected layers for classification
    dense = Dense(units=128, activation='relu')(flatten)
    output = Dense(units=10, activation='softmax')(dense)
    
    # Construct the model
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])