import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda, LayerNormalization, Activation, Add, Concatenate
from keras.layers import Dense, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Attention branch
    attention_conv = Conv2D(filters=256, kernel_size=(1, 1), padding='same')(input_layer)
    attention_softmax = Activation('softmax')(attention_conv)  # Softmax to generate weights
    
    # Contextual processing with attention weights
    weighted_input = Conv2D(filters=256, kernel_size=(1, 1), padding='same', name='weighted_input')(attention_input)
    weighted_input = Concatenate()([attention_input, weighted_input])  # Concatenate original input with weighted
    
    # Reduce dimensionality using another 1x1 convolution
    reduce_layer = Conv2D(filters=128, kernel_size=(1, 1), padding='same')(input_layer)
    reduce_layer = Lambda(lambda x: x[:, :, ::2, ::2])(reduce_layer)  # Reduce spatial dimensions by a factor of 2
    reduce_layer = LayerNormalization(epsilon=1e-6)(reduce_layer)
    reduce_layer = Activation('relu')(reduce_layer)
    
    # Restore dimensionality with another 1x1 convolution
    expanded_layer = Conv2D(filters=256, kernel_size=(1, 1), padding='same')(reduce_layer)
    expanded_layer = Lambda(lambda x: x[:, :, ::2, ::2])(expanded_layer)  # Restore spatial dimensions
    
    # Add processed input to original
    output_layer = Add()([weighted_input, expanded_layer])
    
    # Flatten and fully connected layers for classification
    flattened = Flatten()(output_layer)
    dense = Dense(units=256, activation='relu')(flattened)
    output = Dense(units=10, activation='softmax')(dense)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()