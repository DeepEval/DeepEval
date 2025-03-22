from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose
from keras.models import Model
from keras.applications import VGG16


input_layer = Input(shape=(32, 32, 3))


conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu')(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


conv4 = Conv2D(256, (3, 3), activation='relu')(pool3)
dropout = Dropout(0.2)(conv4)
conv5 = Conv2D(256, (3, 3), activation='relu')(dropout)


upsample1 = Conv2DTranspose(64, (3, 3), activation='relu')(conv5)
upsample2 = Conv2DTranspose(32, (3, 3), activation='relu')(upsample1)
upsample3 = Conv2DTranspose(16, (3, 3), activation='relu')(upsample2)


output_layer = Conv2D(10, (1, 1), activation='softmax')(upsample3)


model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])