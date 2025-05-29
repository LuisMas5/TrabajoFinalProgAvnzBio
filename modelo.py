import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Cargar y preparar datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Configurar pruning
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=(len(x_train) // 128) * 3  # pasos = epochs * steps_per_epoch
    )
}

# Crear modelo pruned
def create_pruned_model():
    model = Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        tfmot.sparsity.keras.prune_low_magnitude(Conv2D(32, (3, 3), activation='relu'), **pruning_params),
        MaxPooling2D((2, 2)),
        Flatten(),
        tfmot.sparsity.keras.prune_low_magnitude(Dense(128, activation='relu'), **pruning_params),
        tfmot.sparsity.keras.prune_low_magnitude(Dense(10, activation='softmax'), **pruning_params)
    ])
    return model

# Crear y compilar el modelo
model = create_pruned_model()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callback necesario para pruning
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
]

# Entrenar
model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=3,
    validation_data=(x_test, y_test),
    callbacks=callbacks
)

# Eliminar capas de pruning para guardar el modelo limpio
model_stripped = tfmot.sparsity.keras.strip_pruning(model)
model_stripped.save("modelo_pruned_final.h5")

print("Entrenamiento y pruning finalizados. Modelo guardado como modelo_pruned_final.h5.")


