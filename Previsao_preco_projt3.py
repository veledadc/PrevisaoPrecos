import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers
print(tf.__version__)

import ssl
ssl._create_default_https_context = ssl._create_default_https_context
dataset_path = keras.utils.get_file('housing.data', 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data')

nomes_colunas = ['CRIM',
                 'ZN',
                 'INDUS',
                 'CHAS',
                 'NOX',
                 'RM',
                 'AGE',
                 'DIS', 
                 'RAD', 
                 'TAX', 
                 'PTRATIO', 
                 'B', 
                 'LSTAT', 
                 'MEDV'] 

dataset = pd.read_csv(
    dataset_path, 
    names = nomes_colunas, 
    na_values = '?', 
    comment = '\t',
    sep = ' ',
    skipinitialspace = True)

print(dataset.head())

dados_treino = dataset.sample(frac=0.8, random_state=0)
dados_teste = dataset.drop(dados_treino.index)

fig, ax = plt.subplots()
x = dados_treino['RM']
y = dados_treino['MEDV']
ax.scatter(x, y, edgecolors = (0, 1, 0))
ax.set_xlabel('RM')
ax.set_xlabel('MEDV')
plt.show()

x_treino = dados_treino['RM']
y_treino = dados_treino['MEDV']
x_teste = dados_teste['RM']
y_teste = dados_teste['MEDV']

def modelo_linear():
    model = keras.Sequential([layers.Dense(1, use_bias = True, input_shape = (1,), name = 'layer' ) ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, 
                                        beta_1=0.9,
                                        beta_2=0.99, 
                                        epsilon=1e-05, 
                                        name='Adam')
    
    model.compile(loss= 'mse', 
                 optimizer = optimizer, 
                 metrics= ['mae', 'mse'])
    
    return model 


modelo = modelo_linear()

tf.keras.utils.plot_model(modelo,
                         to_file= 'modelo.png',
                         show_shapes = True, 
                         show_layer_names = True, 
                         rankdir = 'TB', 
                         expand_nested = False, 
                         dpi = 100)



n_epochs = 4000
batch_size = 256
n_idle_epochs = 100
n_epochs_log = 200
n_sample_save = n_epochs_log * x_treino.shape[0]
print('Checkpoint salvo a cada {} amostra'.format(n_sample_save))

earlyStopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                               patience = n_idle_epochs, 
                                               min_delta = 0.001)


preditctions_list = []

checkpoint_path = 'dados/'

checkpointCallback = tf.keras.callbacks.ModelCheckpoint(filepath='dados/', 
                                                        verbose = 1, 
                                                       save_weights_only= True, 
                                                       save_freq= n_sample_save)


modelo.save_weights(checkpoint_path.format(epoch = 0))

history = modelo.fit(x_treino, 
                    y_treino, 
                    batch_size = batch_size,
                    epochs= n_epochs, 
                    validation_split= 0.1, 
                    verbose = 1, 
                    callbacks= [earlyStopping, checkpointCallback])


print('Keys:', history.history.keys())

mse = np.asarray(history.history['mse'])
val_mse = np.asarray(history.history['val_mse'])

num_values = (len(mse))
values = np.zeros((num_values, 2), dtype = float)
values[:, 0] = mse
values[:, 1] = val_mse

steps = pd.RangeIndex(start = 0, stop = num_values)
df = pd.DataFrame(values, steps, columns=['MSE em treino', 'MSE em validação'])

sns.set(style = 'whitegrid')
sns.lineplot(data = df, palette='tab10', linewidth = 2.5)

previsoes = modelo.predict(y_teste).flatten()


print(previsoes)





