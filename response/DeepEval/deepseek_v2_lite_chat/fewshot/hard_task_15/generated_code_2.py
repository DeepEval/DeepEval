import keras
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_pool = GlobalAveragePooling2D()(main_conv)
    main_dense1 = Dense(units=128, activation='relu')(main_pool)
    main_dense2 = Dense(units=64, activation='relu')(main_dense1)
    
    # Branch path
    branch_conv = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch_dense = Dense(units=64, activation='relu')(branch_conv)
    
    # Combine main and branch paths
    combined = Add()([main_dense2, branch_dense])
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(combined)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()