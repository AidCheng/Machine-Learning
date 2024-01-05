import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from Linear_Regression import LinearRegression

data = pd.read_csv('Linear_Regression/data/world-happiness-report-2017.csv')


# retrieve data for training and test
train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)
breakpoint()

input_x_name = 'Economy..GDP.per.Capita.'
input_y_name = 'Freedom'
output_z_name = 'Happiness.Score'

train_param = train_data[[input_x_name,input_y_name]].values
train_result = train_data[[output_z_name]].values

test_param = test_data[[input_x_name,input_y_name]].values
test_result = test_data[[output_z_name]].values

num_iteration = 500
learning_rate = 0.01
polynomial = 0
sinusoids = 0

linear_regression = LinearRegression(train_param, train_result, polynomial, sinusoids)

(theta, cost_history) = linear_regression.train(
    learning_rate, 
    num_iteration
)

print(theta)

print('beginning cost:', cost_history[0])
print('Eventual cost:', cost_history[-1])

plot_training_trace = go.Scatter3d(
    x = train_param[:,0].flatten(),
    y = train_param[:,1].flatten(),
    z = train_result.flatten(),
    name = 'Training Set',
    mode = 'markers',
    marker = {
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        }
    }
)

plot_layout = go.Layout (
    title = 'Data Sets',
    scene = {
        'xaxis':{'title': input_x_name},
        'yaxis':{'title': input_y_name},
        'zaxis':{'title': output_z_name}
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

predictions_num = 10

x_min = train_param[:,0].min()
x_max = train_param[:,0].max()

y_min = train_param[:,1].min()
y_max = train_param[:,1].max()

x_axis = np.linspace(x_min,x_max,predictions_num)
y_axis = np.linspace(y_min,y_max,predictions_num)

x_param = np.zeros((predictions_num ** 2, 1))
y_param = np.zeros((predictions_num ** 2, 1))

idx = 0
for x_values in x_axis:
    for y_values in y_axis:
        y_param[idx] = y_values
        x_param[idx] = x_values
        idx += 1

params = np.hstack((x_param,y_param))
z_predict_result = linear_regression.predict(params)

plot_predictions_trace = go.Scatter3d(
    x=x_param.flatten(),
    y=y_param.flatten(),
    z=z_predict_result.flatten(),
    name='Prediction Plane',
    mode='markers',
    marker={
        'size': 1,
    },
    opacity=0.8,
    surfaceaxis=2, 
)

plt.plot(range(num_iteration),cost_history)
plt.xlabel('Iter')
plt.ylabel('loss')
plt.title('GD')
plt.show()

plot_data = [plot_training_trace, plot_predictions_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)
plotly.offline.plot(plot_figure)



