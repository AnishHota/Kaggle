import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
iris_data = pd.read_csv('iris.csv')
'''sb.pairplot(iris_data,hue='Species')
sb.plt.show()'''
print iris_data[iris_data['Sepal.Width']<2.5]
