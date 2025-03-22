import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution, 3x3 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    dropout1 = Dropout(rate=0.5)(conv2)
    
    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv4 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='valid', activation='relu')(conv3)
    conv5 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='valid', activation='relu')(conv4)
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    dropout2 = Dropout(rate=0.5)(conv6)
    
    # Branch 3: Max Pooling
    pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    dropout3 = Dropout(rate=0.5)(pool)
    
    # Concatenate outputs from all branches
    concat = Concatenate()(list(map(lambda t: t, [dropout1, dropout2, dropout3])))
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])