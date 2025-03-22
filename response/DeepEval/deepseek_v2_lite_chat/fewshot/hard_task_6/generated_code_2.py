import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.layers.merge import Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path with repeated Block 1
    def block1(x):
        x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
        return x
    
    x = block1(input_layer)
    
    for _ in range(3):  # Repeat Block 1 three times
        x = block1(x)
    
    # Branch path with average pooling
    branch_layer = AveragePooling2D(pool_size=3, strides=1, padding='same')(input_layer)
    
    # Merge the main path and branch path
    merged = Concatenate()([x, branch_layer])
    
    # Add fully connected layers
    x = Flatten()(merged)
    x = Dense(units=128, activation='relu')(x)
    output = Dense(units=10, activation='softmax')(x)
    
    # Construct the model
    model = Model(inputs=[input_layer, branch_layer], outputs=output)
    
    return model

model = dl_model()
model.summary()