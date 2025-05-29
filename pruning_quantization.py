import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

# Carga MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    return model

model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, validation_split=0.1)

_, baseline_model_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Precisión modelo base: {baseline_model_accuracy:.4f}")

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
batch_size = 128
epochs = 5
validation_split = 0.1

num_images = x_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                             final_sparsity=0.5,
                                                             begin_step=0,
                                                             end_step=end_step)
}

pruned_model = prune_low_magnitude(create_model(), **pruning_params)

pruned_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

pruned_model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_split=validation_split,
                 callbacks=callbacks)

_, pruned_model_accuracy = pruned_model.evaluate(x_test, y_test, verbose=0)
print(f"Precisión modelo podado: {pruned_model_accuracy:.4f}")

final_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
final_pruned_model.save("pruned_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(final_pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

with open("quantized_model.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("Modelo cuantizado guardado como quantized_model.tflite")
