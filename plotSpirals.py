import matplotlib.pyplot as plt


xw, yw, xb, yb = [], [], [], []
with open('./two-spiral.txt', 'r') as f:
    for line in f:
        l = line.split()
        if(l[2] == '0.0'):
            xb.append(float(l[0]))
            yb.append(float(l[1]))
        else:
            xw.append(float(l[0]))
            yw.append(float(l[1]))

fig, ax = plt.subplots()
ax.scatter(xb, yb, c='k')
ax.scatter(xw, yw, c='r')
plt.show()
