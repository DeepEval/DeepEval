import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch path for average pooling
    branch_input = MaxPooling2D(pool_size=(4, 4))(input_layer)
    
    # Main path for the model
    def block(input_tensor, num_filters):
        conv = Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
        conv = Conv2D(num_filters, (3, 3), padding='same')(conv)
        conv = Conv2D(num_filters, (5, 5), padding='same', activation='relu')(conv)
        return conv

    # Blocks 1, 2, 3
    num_filters = 32
    block1_output = block(input_tensor=input_layer, num_filters=num_filters)
    num_filters *= 2
    block2_output = block(input_tensor=input_layer, num_filters=num_filters)
    num_filters *= 2
    block3_output = block(input_tensor=input_layer, num_filters=num_filters)
    
    # Concatenate outputs from Blocks
    fused_features = Concatenate()([block1_output, block2_output, block3_output])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(fused_features)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=[input_layer, branch_input], outputs=output)
    
    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])