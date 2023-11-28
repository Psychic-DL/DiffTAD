import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25

# set height of bar
bars1 = [73.17, 63.07, 58.28]
bars2 = [69.88, 58.62, 52.34]

figsize = 8,6
figure, ax = plt.subplots(figsize=figsize)

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

label_font = {
              'weight': 'normal',
              'size': 19}
plt.bar(r1, bars1, color='#5bc49f', width=barWidth, edgecolor='white', label='w. decaying mechanism')
plt.bar(r2, bars2, color='#60acfc', width=barWidth, edgecolor='white', label='w/o. decaying mechanism')

# plt.xlabel('Group',label_font)
plt.ylabel("Accuracy", label_font)
plt.ylim([50, 75])
plt.xticks([r + barWidth for r in range(len(bars1))], ['Candi#3', 'Candi#5', 'Candi#7'])
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontsize(19) for label in labels]

# Create legend & Show graphic
plt.legend(prop=label_font)
plt.savefig("tt.pdf")