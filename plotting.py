import json
import matplotlib.pyplot as plt

with open('output.json', 'r', encoding='utf-8') as file:
    output = json.load(file)

clear = [output[str(i)]['clearance'] for i in range(500)]
fig, ax = plt.subplots(1,1)
ax.hist(clear, bins=250)
ax.grid()
plt.show()