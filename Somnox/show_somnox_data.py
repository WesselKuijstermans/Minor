import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors

df = pd.read_csv('testfile_somnox_labeled.csv')
df.head()
# expirement starts at around 52000 and ends at around 450000
# from 68580 to 373384 is very usable data.
filtered = df.drop(df[df['time'] < 68580].index)
filtered = filtered.drop(filtered[filtered['time'] > 373384].index)

fig, ax = plt.subplots()
fig.set_size_inches(25, 8)
line, = ax.plot(filtered['time'], filtered['somnox_ay'], label='somnox_ay')
# Make it so i can hover over the line and it shows the value of the line at that point in time

plt.xlabel('Time')
plt.ylabel('somnox_ay')
plt.legend()
cursor = mplcursors.cursor(line, hover=True)

# # Format the annotation to show (x, y) values
@cursor.connect("add")
def on_hover(sel):
    sel.annotation.set_text(f"Time: {sel.target[0]:.2f}\nValue: {sel.target[1]:.2f}")
    
plt.isinteractive()
plt.show()