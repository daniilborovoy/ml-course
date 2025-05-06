import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use Agg backend which doesn't require GUI
import matplotlib.pyplot as plt


class DiscriminantAnalysis:
    def __init__(self, filename):
        self.filename = filename
        self.x1 = self.load_data(filename + "1.txt")
        self.x2 = self.load_data(filename + "2.txt")
        self.x0 = self.load_data(filename + "0.txt")

        self.n1 = self.x1.shape[0]
        self.n2 = self.x2.shape[0]
        self.n0 = self.x0.shape[0]
        self.m = self.x0.shape[1]

        self.x11 = np.array(
            [np.mean(self.x1[:, 0]), np.mean(self.x1[:, 1]), np.mean(self.x1[:, 2])]
        )
        self.x21 = np.array(
            [np.mean(self.x2[:, 0]), np.mean(self.x2[:, 1]), np.mean(self.x2[:, 2])]
        )

        self.s1 = self._calculate_covariance_matrix(self.x1, self.x11, self.n1)
        self.s2 = self._calculate_covariance_matrix(self.x2, self.x21, self.n2)

        self.s = (1 / (self.n1 + self.n2 - 2)) * (self.n1 * self.s1 + self.n2 * self.s2)
        self.s0 = np.linalg.inv(self.s)

        self.a = np.dot(self.s0, self.x11 - self.x21)
        self.f1 = np.dot(self.x1, self.a)
        self.f2 = np.dot(self.x2, self.a)

        self.m1 = np.mean(self.f1)
        self.m2 = np.mean(self.f2)
        self.f = (1 / 2) * (self.m1 + self.m2)

        self.f0 = np.dot(self.x0, self.a)
        self.res = self.f0 - self.f
        self.classification = self._classify_points()

    @staticmethod
    def load_data(filename):
        return np.loadtxt(filename)

    def _calculate_covariance_matrix(self, data, means, n):
        tmp = []
        for j in range(self.m):
            for l in range(self.m):
                x = 0
                for i in range(n):
                    x += (data[i][j] - means[j]) * (data[i][l] - means[l])
                x = (1 / n) * x
                tmp.append(x)
        return np.reshape(tmp, (self.m, self.m))

    def _classify_points(self):
        classification = []
        if self.m1 > self.m2:
            for i in range(len(self.res)):
                classification.append("X1" if self.res[i] > 0 else "X2")
        else:
            for i in range(len(self.res)):
                classification.append("X2" if self.res[i] > 0 else "X1")
        return classification

    def plot_results(self):
        ff = np.concatenate(
            (self.f1 - self.f, self.f2 - self.f, self.f0 - self.f), axis=0
        )

        len1 = len(self.f1)
        len2 = len(self.f2)
        len3 = len(self.f0)

        x1 = np.arange(0, len1)
        x2 = np.arange(len1, len1 + len2)
        x3 = np.arange(len1 + len2, len1 + len2 + len3)

        plt.figure(figsize=(10, 6))
        plt.plot(x1, self.f1 - self.f, "o", label="f1", color="green")
        plt.plot(x2, self.f2 - self.f, "o", label="f2", color="red")
        plt.plot(x3, self.f0 - self.f, "o", label="f0", color="blue")

        plt.axhline(0, color="pink", linewidth=1.5, label="x = 0")

        plt.title("Распределение точек")
        plt.xlabel("Порядковый номер элемента")
        plt.ylabel("Значение")
        plt.grid(True)
        plt.legend()
        plt.savefig("discriminant_analysis_plot.png")
        plt.close()


if __name__ == "__main__":
    filename = input("Введите название файла: ")
    da = DiscriminantAnalysis(filename)
    print(da.res)
    print(da.classification)
    da.plot_results()
