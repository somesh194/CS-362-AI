import matplotlib.pyplot as plt
class graphing():
    def __init__(self, coordinates, place, best_path):
        self.coordin = coordinates
        self.place = place
        self.best_path=best_path

    def grapf(self):
        k=len(self.best_path)
        x=[0]*(k+1)
        y=[0]*(k+1)
        for i in range(0,k):
                x[i] = self.coordin[self.best_path[i]][0]
        for i in range(0,k):
                y[i] = self.coordin[self.best_path[i]][1]
        x[k]=self.coordin[self.best_path[0]][0]
        y[k]=self.coordin[self.best_path[0]][1]
        plt.plot(x, y, color='green', linestyle='dashed', linewidth = 1,marker='o', markerfacecolor='blue', markersize=2)
        annotations=[""]*(k+1)
        for i in range(0,k):
            o = self.best_path[i]
            annotations[i] = self.place[o]
        annotations[k]=self.place[self.best_path[0]]
        for i, label in enumerate(annotations):
            plt.annotate(label, (x[i], y[i]))
        plt.show()