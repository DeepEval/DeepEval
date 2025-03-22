import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, Layer, Activation, Concatenate, Add, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import LayerNormalization, ReLU, Softmax

# Load and preprocess CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Function to create the attention mechanism
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W', shape=(input_shape[3], 1),
                                 initializer='uniform', trainable=True)
        self.b = self.add_weight(name='b', shape=(input_shape[3], 1),
                                 initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        a = K.sigmoid(K.dot(x, self.W) + self.b)
        return x * a

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], input_shape[3]

# Define the model
def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Attention mechanism
    attention = AttentionLayer()(input_layer)
    
    # Reduce dimensionality and apply ReLU
    attention = Conv2D(filters=int(input_layer.shape[3]/3), kernel_size=1, activation='relu')(attention)
    
    # Restore dimensionality
    attention = Conv2D(filters=input_layer.shape[3], kernel_size=1, activation='sigmoid')(attention)
    
    # Concatenate the attention-weighted feature with the original input
    concatted = Concatenate()([input_layer, attention])
    
    # Flatten and process for classification
    flat = Flatten()(concatted)
    output = Dense(10, activation='softmax')(flat)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create and compile the model
model = dl_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)