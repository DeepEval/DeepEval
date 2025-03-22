import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Max Pooling Layers
    maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    
    # Flatten and concatenate
    flat1 = Flatten()(maxpool1)
    flat2 = Flatten()(maxpool2)
    flat3 = Flatten()(maxpool3)
    
    concat = Concatenate()([flat1, flat2, flat3])
    
    # Fully connected layers
    dense1 = Dense(units=1024, activation='relu')(concat)
    dense2 = Dense(units=512, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes in CIFAR-10
    
    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()