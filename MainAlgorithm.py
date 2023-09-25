import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class NQueens:

    def __init__(self, numOfQueens):
        # инициализация переменных
        self.numOfQueens = numOfQueens

    def __len__(self):
        # кол-во ферзей
        return self.numOfQueens

    def getViolationsCount(self, positions):
        # Вычисляет количество нарушений в данном решении
        # Поскольку входные данные содержат уникальные индексы
        # столбцов для каждой строки, нарушения строк или столбцов невозможны,
        # необходимо учитывать только диагональные нарушения.

        if len(positions) != self.numOfQueens:
            raise ValueError("Размер списка позиций должен быть равен ", self.numOfQueens)

        violations = 0

        # перебираем каждую пару ферзей и проверяем, находятся ли они на одной диагонали:
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):

                column1 = i
                row1 = positions[i]

                column2 = j
                row2 = positions[j]

                if abs(column1 - column2) == abs(row1 - row2):
                    violations += 1

        return violations

    def plotBoard(self, positions):
        # Отображает положение ферзей на доске в соответствии с заданным решением

        if len(positions) != self.numOfQueens:
            raise ValueError("Размер списка позиций должен быть равен ", self.numOfQueens)

        fig, ax = plt.subplots()

        # строим шахматную доску
        board = np.zeros((self.numOfQueens, self.numOfQueens))

        board[::2, 1::2] = 1
        board[1::2, ::2] = 1

        ax.imshow(board, interpolation='none', cmap=mpl.colors.ListedColormap(['#ffce9e', '#d18b47']))

        # отображаем ферзя:
        queenThumbnail = plt.imread('queens.png')
        thumbnailSpread = 0.70 * np.array([-1, 1, -1, 1]) / 2

        # перебираем позиции ферзя - i - строка, j - столбец:
        for i, j in enumerate(positions):
            ax.imshow(queenThumbnail, extent=[j, j, i, i] + thumbnailSpread)

        # отображение индексов строк и столбцов:
        ax.set(xticks=list(range(self.numOfQueens)), yticks=list(range(self.numOfQueens)))

        ax.axis('image')

        return plt


def main():
    # создание экземпляра
    nQueens = NQueens(8)

    # заведомо правильное решение:
    # solution = [5, 0, 4, 1, 7, 2, 6, 3]

    # решение с 3 нарушениями:
    solution = [1, 2, 7, 5, 0, 3, 4, 6]

    print("Число нарушений = ", nQueens.getViolationsCount(solution))

    plot = nQueens.plotBoard(solution)
    plot.show()


if __name__ == "__main__":
    main()