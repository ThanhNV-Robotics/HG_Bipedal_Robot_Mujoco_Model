import json
import matplotlib.pyplot as plt
import numpy as np

# Open and load the JSON file
with open('save_data.json', 'r') as file:
    data = json.load(file)
    data = np.array(data)

time = np.array(data[:,0])

plt.figure()
plt.plot(time, data[:,1], label = "Right Hip Pitch Angle")
plt.plot(time, data[:,2], ':', label = "Right Hip Roll Angle")

plt.legend()
plt.show()