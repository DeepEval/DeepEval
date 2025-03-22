import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply
from keras.models import Model

def dl_model():     
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Define first convolutional branch (3x3)
    branch1_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1_conv1)

    # Step 3: Define second convolutional branch (5x5)
    branch2_conv1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_conv2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch2_conv1)

    # Step 4: Combine branches through addition
    combined = Add()([branch2_conv2, branch1_conv2])

    # Step 5: Global average pooling
    pooled = GlobalAveragePooling2D()(combined)

    # Step 6: Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(pooled)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Step 7: Generate attention weights
    attention_weights = Dense(units=10, activation='softmax')(dense2)

    # Step 8: Compute the final output (weighted output)
    final_output = Multiply()([pooled, attention_weights])

    # Step 9: Fully connected layer for the final classification
    output_layer = Dense(units=10, activation='softmax')(final_output)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create the model
model = dl_model()
model.summary()