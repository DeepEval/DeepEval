import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Main pathway
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    drop1 = Dropout(0.5)(pool1)  # 50% dropout to mitigate overfitting
    
    # Branch pathway
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop1)
    
    # Fusion of main and branch pathways
    fused_output = Add()([pool1, branch_conv])
    
    # Additional blocks
    bath_norm = BatchNormalization()(fused_output)
    flatten_layer = Flatten()(bath_norm)
    
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])