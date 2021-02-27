import mxnet as mx
import numpy as np

import logging
logging.getLogger().setLevel(logging.DEBUG)

#Training data
train_data = np.loadtxt("housing.data", max_rows=405, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
train_label = np.loadtxt("housing.data", max_rows=405, usecols=13)
batch_size = 1

#Evaluation Data
eval_data = np.loadtxt("housing.data", skiprows=405, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
eval_label = np.loadtxt("housing.data", skiprows=405, usecols=13)

train_iter = mx.io.NDArrayIter(train_data,train_label, batch_size, shuffle=True,label_name='lin_reg_label')
eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)

X = mx.sym.Variable('data')
Y = mx.symbol.Variable('lin_reg_label')
fully_connected_layer  = mx.sym.FullyConnected(data=X, name='fc1', num_hidden = 1)
lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")

model = mx.mod.Module(
    symbol = lro ,
    data_names=['data'],
    label_names = ['lin_reg_label']# network structure
)

mx.viz.plot_network(symbol=lro)

model.fit(train_iter, eval_iter,
            optimizer_params={'learning_rate':0.005, 'momentum': 0.9},
            num_epoch=20,
            eval_metric='mse',
            batch_end_callback = mx.callback.Speedometer(batch_size, 2))

model.predict(eval_iter).asnumpy()

metric = mx.metric.MSE()
model.score(eval_iter, metric)
assert model.score(eval_iter, metric)[0][1] < 0.01001, "Achieved MSE (%f) is larger than expected (0.01001)" % model.score(eval_iter, metric)[0][1]

eval_data = np.array([[7,2],[6,10],[12,2]])
eval_label = np.array([11.1,26.1,16.1]) #Adding 0.1 to each of the values
eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)
model.score(eval_iter, metric)