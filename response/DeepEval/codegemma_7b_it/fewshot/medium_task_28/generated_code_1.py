import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Multiply, Reshape, LayerNormalization

def attention_block(input_tensor):
    attention = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(input_tensor)
    return attention

def context_block(input_tensor, attention):
    context = Multiply()([input_tensor, attention])
    return context

def reduce_dim(input_tensor):
    reduced = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    return reduced

def restore_dim(input_tensor):
    restored = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    return restored

def residual_block(input_tensor):
    reduced = reduce_dim(input_tensor)
    normalized = LayerNormalization()(reduced)
    activated = keras.layers.ReLU()(normalized)
    restored = restore_dim(activated)
    return added

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    attention = attention_block(input_layer)
    context = context_block(input_layer, attention)
    added = keras.layers.Add()([input_layer, context])
    reshaped = Reshape((32, 32, -1))(added)
    residually_processed = residual_block(reshaped)
    flatten_layer = Flatten()(residually_processed)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])