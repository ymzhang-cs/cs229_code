import numpy as np
import matplotlib.pyplot as plt


class LMS:
    def __init__(self, data: np.array, init_param, learning_rate, cost_threshold, time_limit):
        self.data = data
        self.learning_rate = learning_rate
        self.cost_threshold = cost_threshold
        self.time_limit = time_limit
        # 初始化参数
        if isinstance(init_param, np.ndarray):
            self.param = init_param
        else:
            self.param = np.random.rand(self.data.shape[1])
        # 样本数目
        self.sample_num = self.data.shape[0]
        # 参数数目
        self.param_num = self.data.shape[1]
        # 记录损失函数值
        self.cost_record = np.array([])

    def BGD(self):
        """
        BGD refers to "batch gradient descent".
        :return: None
        """
        cnt = 0
        while self.cost() > self.cost_threshold and cnt <= self.time_limit:
            for i in range(self.param_num):
                self.param[i] += self.LMS_Updated(i)
            cnt += 1
            self.cost_record = np.append(self.cost_record, [self.cost()])
            print(self.cost(), f"theta0 = {self.param[0]}, theta1 = {self.param[1]}")

        print(f"theta0 = {self.param[0]}, theta1 = {self.param[1]}\n"
              f"Get function: y = {self.param[0]} + {self.param[1]} x\n"
              f"with cost function value of {self.cost()}\n"
              f"After {cnt} times training.")

    def LMS_Updated(self, feature_index):
        """
        Calculate LMS_Updated( (y - h) * x ) value for provided feature.
        :param feature_index: feature_index
        :return: (float) LMS_Updated
        """
        sum_value = 0.0
        for sample_i in range(self.sample_num):
            y = self.data[sample_i][-1]
            h = self.hypothesis(sample_i)
            x = self.data[sample_i][feature_index]
            # print(f"y-h={y-h}, y={y}, h={h}")
            sum_value += (y - h) * x
        return self.learning_rate * sum_value

    def hypothesis(self, sample_index):
        """
        Calculate hypothesis value for current sample.
        :param sample_index:
        :return: (float) h value
        """
        # 使用矩阵运算更方便
        result = self.param @ np.insert(self.data[sample_index], 0, 1)[:-1].reshape(-1, 1)
        return result[0]

    def cost(self):
        cost_value = 0.0
        for sample_i in range(self.sample_num):
            y = self.data[sample_i][-1]
            h = self.hypothesis(sample_i)
            cost_value += (y - h) ** 2 / 2
        return cost_value

    def plot_cost_record(self):
        plt.figure()
        plt.plot(self.cost_record)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss Function Evolution')
        plt.savefig('cost_record.jpg')
        plt.show()


if __name__ == '__main__':
    experiment_data = np.genfromtxt('electrical-va-data.csv', delimiter=',', max_rows=2).T
    print(experiment_data)

    test = LMS(experiment_data, init_param=np.array([0.0, 10.0], dtype=float), learning_rate=0.001, cost_threshold=0.001, time_limit=100000)
    test.BGD()
    test.plot_cost_record()
