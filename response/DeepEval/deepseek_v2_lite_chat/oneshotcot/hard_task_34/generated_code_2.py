import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)
        return max_pooling
    
    # Main path
    main_path = []
    for _ in range(3):  # Repeat the block three times
        main_path.append(block(input_tensor=input_layer))
    
    # Branch path
    branch_input = main_path[0]
    for _ in range(3):  # Repeat the branch convolutional layer three times
        branch_input = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_input)
    
    # Concatenate the feature maps from both paths
    concat = Concatenate()(main_path + [branch_input])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])