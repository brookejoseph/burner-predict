import numpy as np
import pandas as pd

np.random.seed(42)
data_size = 1000
fuel_flow = np.random.uniform(50, 100, data_size)
air_pressure = np.random.uniform(30, 50, data_size)
burner_flame = np.random.uniform(800, 1200, data_size)

burner_temperature = 0.5 * fuel_flow + 0.2 * air_pressure + 0.3 * burner_flame + np.random.normal(0, 5, data_size)
emissions = 0.1 * burner_temperature + 0.05 * fuel_flow + np.random.normal(0, 1, data_size)

df = pd.DataFrame({
    'Fuel Flow': fuel_flow,
    'Air Pressure': air_pressure,
    'Burner Flame': burner_flame,
    'Burner Temperature': burner_temperature,
    'Emissions': emissions
})

from sklearn.model_selection import train_test_split

X = df[['Fuel Flow', 'Air Pressure', 'Burner Flame']]
y_temperature = df['Burner Temperature']
y_emissions = df['Emissions']

X_train, X_test, y_temp_train, y_temp_test, y_em_train, y_em_test = train_test_split(
    X, y_temperature, y_emissions, test_size=0.2, random_state=42
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

mlp_temp = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

mlp_temp.compile(optimizer='adam', loss='mse')

mlp_em = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

mlp_em.compile(optimizer='adam', loss='mse')

mlp_temp.fit(X_train, y_temp_train, epochs=50, batch_size=32, verbose=1)
mlp_em.fit(X_train, y_em_train, epochs=50, batch_size=32, verbose=1)



def incremental_update(model, X_new, y_new, batch_size=32, epochs=1):
    X_new = X_new.reshape(X_new.shape[1], X_new.shape[0]) # Transpose
    model.fit(X_new, y_new, batch_size=batch_size, epochs=epochs, verbose=0)

new_data_size = 100
new_fuel_flow = np.random.uniform(50, 100, new_data_size)
new_air_pressure = np.random.uniform(30, 50, new_data_size)
new_burner_flame = np.random.uniform(800, 1200, new_data_size)

X_new = np.column_stack([new_fuel_flow, new_air_pressure, new_burner_flame])

y_new_temp




def virtual_sensor(X):
    return mlp_em.predict(X)

predicted_emissions = virtual_sensor(X_test)
print(f"Predicted emissions: {predicted_emissions[:5]}")



from scipy.optimize import minimize

def cost_function(params):
    fuel_flow, air_pressure, burner_flame = params
    X_input = np.array([[fuel_flow, air_pressure, burner_flame]])
    predicted_temp = mlp_temp.predict(X_input)
    predicted_em = mlp_em.predict(X_input)
    return predicted_em + abs(predicted_temp - target_temp)

target_temp = 950

result = minimize(cost_function, x0=[60, 40, 1000], bounds=[(50, 100), (30, 50), (800, 1200)])
optimal_params = result.x
print(f"Optimal Burner Parameters: {optimal_params}")



