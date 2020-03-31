from math import sqrt
from random import randint
import numpy as np
import scipy.stats


def r(x: float) -> float:
    """Точність округлення"""
    x = round(x, 4)
    if float(x) == int(x):
        return int(x)
    else:
        return x


def par(a: float) -> str:
    """Для вивіду. Негативні числа закидає в скобки и округлює"""
    if a < 0:
        return "(" + str(r(a)) + ")"
    else:
        return str(r(a))


def average(list: list, name: str) -> int or float:
    """Середнє значення з форматованним вивідом для будь-якого листа"""
    print("{} = ( ".format(name), end="")
    average = 0
    for i in range(len(list)):
        average += list[i]
        if i == 0:
            print(r(list[i]), end="")
        else:
            print(" + ", end="")
            print(par(list[i]), end="")
    average /= len(list)
    print(" ) / {} = {} ".format(len(list), r(average)))
    return average


def printf(name: str, value: int or float):
    """Форматованний вивід змінної з округленням"""
    print("{} = {}".format(name, r(value)))


def matrixplan(xn_factor: list, x_min: int, x_max: int) -> list:
    """Заповнює матрицю планування згідно нормованої"""
    xn_factor_experiment = []
    for i in range(len(xn_factor)):
        if xn_factor[i] == -1:
            xn_factor_experiment.append(x_min)
        elif xn_factor[i] == 1:
            xn_factor_experiment.append(x_max)
    return xn_factor_experiment


def MatrixExper(x_norm: list, x_min: list, x_max: list) -> list:
    """Генеруємо матрицю планування згідно нормованної"""
    x_factor_experiment = []
    for i in range(len(x_norm)):
        if i == 0:
            continue  # x0_factor = [1, 1, 1, 1] не використовується на першому этапы
            # стовпці матриці планування (эксперементальні)
        x_factor_experiment.append(matrixplan(x_norm[i], x_min[i - 1], x_max[i - 1]))
    return x_factor_experiment


def generate_y(y_min, y_max, n, m) -> list:
    """Генерує функції відгугу за вказанним діапозоном"""
    list = []
    for i in range(m):
        list.append([randint(y_min, y_max + 1) for i in range(n)])
    return list


def a_n_funct(xn_factor_experiment: list, y_average_list: list) -> list:
    """Рахує а1, а2, а3 з форматованним вивідом"""
    a_n = []
    for i in range(len(xn_factor_experiment)):
        a_n.append(0)
        print("a{} = ( ".format(i + 1), end="")
        for j in range(len(xn_factor_experiment[i])):
            a_n[i] += xn_factor_experiment[i][j] * y_average_list[j]
            if j == 0:
                print("{}*{}".format(r(xn_factor_experiment[i][j]), par(y_average_list[j])), end="")
            else:
                print(" + {}*{}".format(par(xn_factor_experiment[i][j]), par(y_average_list[j])), end="")
        a_n[i] /= len(xn_factor_experiment[i])
        print(" ) / {} = {} ".format(len(xn_factor_experiment[i]), r(a_n[i])))
    return a_n


def a_nn_funct(xn_factor_experiment: list) -> list:
    """Рахує а11, а22, а33 з форматованим вивідом"""
    a_nn = []
    for i in range(len(xn_factor_experiment)):
        a_nn.append(0)
        print("a{}{} = ( ".format(i + 1, i + 1), end="")
        for j in range(len(xn_factor_experiment[i])):
            a_nn[i] += xn_factor_experiment[i][j] ** 2
            if j == 0:
                print("{}^2".format(par(xn_factor_experiment[i][j])), end="")
            else:
                print(" + {}^2".format(par(xn_factor_experiment[i][j])), end="")
        a_nn[i] /= len(xn_factor_experiment[i])
        print(" ) / {} = {} ".format(len(xn_factor_experiment[i]), r(a_nn[i])))
    return a_nn


def a_mn_funct(x_factor_experiment: list) -> list:
    """Рахує a12, a21, a13, a31, a23, a32"""
    a_mn = []
    list_range = [[0, 1], [1, 2], [2, 0]]

    for i, j in list_range:
        a_mn.append(0)
        print("a{}{} = ( ".format(i + 1, j + 1), end="")
        for k in range(len(x_factor_experiment[i])):
            a_mn[i] += x_factor_experiment[i][k] * x_factor_experiment[j][k]
            if k == 0:
                print("{}*{}".format(r(x_factor_experiment[i][k]), par(x_factor_experiment[j][k])), end="")
            else:
                print(" + {}*{}".format(r(x_factor_experiment[i][k]), par(x_factor_experiment[j][k])), end="")
        a_mn[i] /= len(x_factor_experiment[i])
        print(" ) / {} = {} ".format(len(x_factor_experiment[i]), r(a_mn[i])))

    return a_mn


def dispers(y: list, y_average_list: list, m) -> list:
    """Рахує s2 для усіх рядків. Повертає масив значень"""
    s2_y_row = []

    for i in range(len(y_average_list)):
        s2_y_row.append(0)
        print("s2_y_row{} = ( ".format(i + 1), end="")
        for j in range(3):
            s2_y_row[i] += (y[j][i] - y_average_list[i]) ** 2
            if j == 0:
                print("({} - {})^2".format(r(y[j][i]), par(y_average_list[i])), end="")
            else:
                print(" + ({} - {})^2".format(r(y[j][i]), par(y_average_list[i])), end="")
        s2_y_row[i] /=  m
        print(" ) / {} = {} ".format(m, r(s2_y_row[i])))

    return s2_y_row


def beta(x_norm: list, y_average_list: list) -> list:
    """Рахує Бета критерия Стюдента. Повертає масив значень"""
    beta_list = []

    for i in range(len(x_norm)):
        beta_list.append(0)
        print("Beta{} = ( ".format(i + 1), end="")
        for j in range(len(x_norm[i])):
            beta_list[i] += y_average_list[j] * x_norm[i][j]
            if j == 0:
                print("{}*{}".format(r(y_average_list[j]), par(x_norm[i][j])), end="")
            else:
                print(" + {}*{}".format(r(y_average_list[j]), par(x_norm[i][j])), end="")
        beta_list[i] /= len(x_norm[0])
        print(" ) / {} = {} ".format(len(x_norm[0]), r(beta_list[i])))

    return beta_list


def t(beta_list: list, s_BetaS) -> list:
    """Рахує t критерія Стюдента. Повертає масив значень"""
    t_list = []
    for i in range(len(beta_list)):
        t_list.append(abs(beta_list[i]) / s_BetaS)
        print("t{} = {}/{} = {}".format(i, r(abs(beta_list[i])), par(s_BetaS), par(t_list[i])))
    return t_list


def s2_od_func(y_average_list, y_average_row_Student, m, N, d):
    """Вираховує сігму в квадраті для критерія Фішера"""
    s2_od = 0
    print("s2_od = ( ", end="")
    for i in range(len(y_average_list)):
        s2_od += (y_average_row_Student[i] - y_average_list[i]) ** 2
        if i == 0:
            print("({} - {})^2".format(r(y_average_row_Student[i]), par(y_average_list[i])), end="")
        else:
            print(" + ({} - {})^2".format(r(y_average_row_Student[i]), par(y_average_list[i])), end="")
    s2_od *= m / (N - d)
    print(" ) * {}/({} - {}) = {} ".format(m, N, d, r(s2_od)))
    return s2_od
 

x_min = [-30, 25, 25]  # Задані за умовою значення. Варіант 206
x_max = [20,  45,  30]

x_average_min = average(x_min, "X_average_min")  # Середнє Х макс и мин
x_average_max = average(x_max, "X_average_max")  # Використ. тільки для визначення варіанту

m = 3  # За замовчуванням
q = 0.05  # рівень значимості
y_max = round(200 + x_average_max)  # Максимальні і мінімальні значення для генерації функції відгуку
printf("Y_max", y_max)
y_min = round(200 + x_average_min)
printf("Y_min", y_min)


# Стовпці матриці планування (нормована)
x_norm = [[1, 1, 1, 1],
             [-1, -1, 1, 1],
             [-1, 1, -1, 1],
             [-1, 1, 1, -1]]  # масив нормованої матриці
             
x_factor_experiment = MatrixExper(x_norm, x_min, x_max)  # Формуємо експер. матрицю

while True:
    y = generate_y(y_min, y_max, len(x_norm[0]), m)  # генеруємо значення функції відгуку

    y_average_list = []  # cереднє значення рядка Y
    for i in range(len(y[0])):
        y_average_list.append(average([y[j][i] for j in range(m)], "y_average_{}row".format(i + 1)))

    y_average_average = average(y_average_list, "Y_average_average")  # середнє середніх значень Y

    x_average = []  # cереднє стовпчика Х експеремент
    for i in range(len(x_factor_experiment)):
        x_average.append(average(x_factor_experiment[i], "X{}_average".format(i + 1)))

    a_n = a_n_funct(x_factor_experiment, y_average_list)  # шукаю a1,a2,a3

    a_nn = a_nn_funct(x_factor_experiment)  # шукаю a11,a22,a33

    a_mn = a_mn_funct(x_factor_experiment)  # ierf. a12, a13, a23

    # y = b0 + b1 x1 + b2 x2+ b3 x3

    # Пошук коефицієнтів регресії
    numerator = np.array([[y_average_average, x_average[0], x_average[1], x_average[2]],
                          [a_n[0], a_nn[0], a_mn[0], a_mn[2]],
                          [a_n[1], a_mn[0], a_nn[1], a_mn[1]],
                          [a_n[2], a_mn[2], a_mn[1], a_nn[2]]])

    denominator = np.array([[1, x_average[0], x_average[1], x_average[2]],
                            [x_average[0], a_nn[0], a_mn[0], a_mn[2]],
                            [x_average[1], a_mn[0], a_nn[1], a_mn[1]],
                            [x_average[2], a_mn[2], a_mn[1], a_nn[2]]])

    b0 = np.linalg.det(numerator) / np.linalg.det(denominator)
    printf("b0", b0)

    numerator = np.array([[1, y_average_average, x_average[1], x_average[2]],
                          [x_average[0], a_n[0], a_mn[0], a_mn[2]],
                          [x_average[1], a_n[1], a_nn[1], a_mn[1]],
                          [x_average[2], a_n[2], a_mn[1], a_nn[2]]])

    denominator = np.array([[1, x_average[0], x_average[1], x_average[2]],
                            [x_average[0], a_nn[0], a_mn[0], a_mn[2]],
                            [x_average[1], a_mn[0], a_nn[1], a_mn[1]],
                            [x_average[2], a_mn[2], a_mn[1], a_nn[2]]])

    b1 = np.linalg.det(numerator) / np.linalg.det(denominator)
    printf("b1", b1)

    numerator = np.array([[1, x_average[0], y_average_average, x_average[2]],
                          [x_average[0], a_nn[0], a_n[0], a_mn[2]],
                          [x_average[1], a_mn[0], a_n[1], a_mn[1]],
                          [x_average[2], a_mn[2], a_n[2], a_nn[2]]])

    denominator = np.array([[1, x_average[0], x_average[1], x_average[2]],
                            [x_average[0], a_nn[0], a_mn[0], a_mn[2]],
                            [x_average[1], a_mn[0], a_nn[1], a_mn[1]],
                            [x_average[2], a_mn[2], a_mn[1], a_nn[2]]])

    b2 = np.linalg.det(numerator) / np.linalg.det(denominator)
    printf("b2", b2)

    numerator = np.array([[1, x_average[0], x_average[1], y_average_average],
                          [x_average[0], a_nn[0], a_mn[0], a_n[0]],
                          [x_average[1], a_mn[0], a_nn[1], a_n[1]],
                          [x_average[2], a_mn[2], a_mn[1], a_n[2]]])

    denominator = np.array([[1, x_average[0], x_average[1], x_average[2]],
                            [x_average[0], a_nn[0], a_mn[0], a_mn[2]],
                            [x_average[1], a_mn[0], a_nn[1], a_mn[1]],
                            [x_average[2], a_mn[2], a_mn[1], a_nn[2]]])

    b3 = np.linalg.det(numerator) / np.linalg.det(denominator)
    printf("b3", b3)
    b_koef = [b0, b1, b2, b3]

    print("Отримане рівняння регресії: y = {} + {}*x1 + {}*x2 + {}*x3".format(r(b0), par(b1), par(b2), par(b3)))

    # Перевірка вірності складеного рівняння
    y_average_row_controls = []
    for i in range(4):
        y_average_row_controls.append(0)
        y_average_row_controls[i] = b0 + b1 * x_factor_experiment[0][i] + b2 * x_factor_experiment[1][i] + b3 * x_factor_experiment[2][i]
        if abs(y_average_row_controls[i] - y_average_list[i]) >= 0.001:
            print(
                "\033[0m Yrow{} = {} + {}*{} + {}*{} + {}*{} = \033[31m {}\t\t\t\033[0mY_average_{}row = \033[31m {}\033[0m".format(
                    i + 1, r(b0), par(b1), par(x_factor_experiment[0][i]), par(b2),
                    par(x_factor_experiment[1][i]), par(b3),
                    par(x_factor_experiment[2][i]),
                    r(y_average_row_controls[i]), i + 1, r(y_average_list[i])))
        else:
            print("Yrow{} = {} + {}*{} + {}*{} + {}*{} = {}\t\t\tY_average_{}row =  {}".format(i + 1, r(b0), par(b1),par(x_factor_experiment[0][i]),
                                                                                              par(b2), par(x_factor_experiment[1][i]), par(b3),
                                                                                              par(x_factor_experiment[2][i]),
                                                                                              r(y_average_row_controls[i]),i + 1,r(y_average_list[i])))
            print("Результат збігається! (точність 0.001)")

    print("Критерія Кохрена")

    s2_list = dispers(y, y_average_list, m)  # дисперсії по рядках

    Gp = max(s2_list) / sum(s2_list)
    print("Gp = (max(s2) / sum(s2)) = {}".format(par(Gp)))
    print("f1=m-1={} ; f2=N=4 Рівень значимості приймемо 0.05.".format(m))
    f1 = m - 1
    f2 = N = 4
    N = 4
    Gt_tableN4 = {1: 0.9065, 2: 0.7679, 3: 0.6841, 4: 0.6287, 5: 0.5892, 6: 0.5598, 7: 0.5365, 8: 0.5175, 9: 0.5017,
                  10: 0.4884}
    Gt = Gt_tableN4[f1]  # табличне значення критерію Кохрена при N=4, f1=2, рівень значимості 0.05
    printf("Gt", Gt)
    if Gp <= Gt:
        Krit_Kohr = "Однор" + " m=" + str(m)
        print("Дисперсія однорідна")
        break
    else:
        Krit_Kohr = "Не однор."
        print("Дисперсія неоднорідна\n\n\n\n")
        print("m+1")
        m += 1
        if m > 10:
            print("Недостатньо інформації для обчислення.")
            quit()

print("Далі оцінимо значимість коефіцієнтів регресії згідно критерію Стьюдента")

s2_B = sum(s2_list) / len(s2_list)
printf("s2_B", s2_B)

s2_BetaS = s2_B / (N * m)
printf("s2_BetaS", s2_BetaS)

s_BetaS = sqrt(s2_BetaS)
printf("s_betaS", s_BetaS)

beta_list = beta(x_norm, y_average_list)  # значенння B0, B1, B2, B3

t_list = t(beta_list, s_BetaS)  # t0, t1, t2, t3

f3 = (m - 1) * N  # N завжди 4
# print("T_tab:", t_tab)
t_tabl = scipy.stats.t.ppf((1 + (1 - q)) / 2, f3)  # табличне значення за критерієм Стюдента
printf("t_tabl", t_tabl)

b_list = []
print("Утворене рівняння регресії: Y = ", end="")
for i in range(len(t_list)):
    b_list.append(0)
    if t_list[i] > t_tabl:
        b_list[i] = b_koef[i]
        if i == 0:
            print("{}".format(r(b_koef[i])), end="")
        else:
            print(" + {}*X{}".format(par(b_koef[i]), i), end="")
print()

# Порівняння результатів
y_average_row_Student = []
dodanki = []
for i in range(4):
    for j in range(len(b_list)):
        if j == 0:
            dodanki.append("{}".format(r(b_list[j])))
        else:
            if b_list[j] == 0:
                dodanki.append("")
            else:
                dodanki.append(" + {}*{}".format(par(b_list[j]), x_factor_experiment[j-1][i]))
    y_average_row_Student.append(0)
    y_average_row_Student[i] = b_list[0] + b_list[1] * x_factor_experiment[0][i] + b_list[2] * x_factor_experiment[1][i] \
                               + b_list[3] * x_factor_experiment[2][i]

    if abs(y_average_row_Student[i] - y_average_list[i]) >= 20:
        print("Yrow{} = {}{}{}{} = \033[31m {}\t\t\t\033[0mY_average_{}row = \033[31m {}\033[0m".format(
            i + 1, dodanki[0], dodanki[1], dodanki[2], dodanki[3],
            r(y_average_row_Student[i]), i + 1, r(y_average_list[i])))
    elif abs(y_average_row_Student[i] - y_average_list[i]) >= 10:
        print("Yrow{} = {}{}{}{} = {}\t\t\tY_average_{}row =  {}".format(
            i + 1, dodanki[0], dodanki[1], dodanki[2], dodanki[3],
            r(y_average_row_Student[i]), i + 1, r(y_average_list[i])))
        print("Результат приблизно (+-10) збігається! (Рівень значимості 0.05)")
    else:
        print("Yrow{} = {}{}{}{} = {}\t\t\tY_average_{}row =  {}".format(
            i + 1, dodanki[0], dodanki[1], dodanki[2], dodanki[3],
            r(y_average_row_Student[i]), i + 1, r(y_average_list[i])))
        print("Результат приблизно (+-10) збігається! (Рівень значимості 0.05)")
    dodanki.clear()
print("Критерій Фішера")
d = b_list.count(0)
f4 = N - d
s2_od = s2_od_func(y_average_list, y_average_row_Student, m, N, d)

Fp = s2_od / s2_B
print("Fp = {} / {} = {}".format(r(s2_od), par(s2_B), r(Fp)))

# Ft = 4.5  # для f3=8; f4=2
F_table = scipy.stats.f.ppf(1 - q, f4, f3)
printf("F_table", F_table)

if Fp > F_table:
    print("За критерієм Фішера рівняння регресії неадекватно оригіналу при рівні значимості 0.05")
    Krit_Fish = "Не адекв."
    
    #ŷ = b0 + b1 x1 + b2 x2 + b3 x3 + b12 x1 x2 + b13 x1 x3 + b23 x2 x3 + b123 x1 x2 x3
    
    # Стовпці матриці планування (нормована)
    x_norm = [[1, 1, 1, 1, 1, 1, 1, 1],
                 [-1, -1, 1, 1, -1, -1, +1, 1],
                 [-1, 1, -1, 1, -1, 1, -1, 1],
                 [-1, 1, 1, -1, 1 ,-1, -1, 1]]  # масив нормованої матриці
    
    x_factor_experiment = MatrixExper(x_norm, x_min, x_max)  # Формуємо експер. матрицю
    x_exper_neight = []
    for j in range(len(x_norm)):
        x_exper_neight.append([])
        for i in range(len(x_factor_experiment[0])):
            if j==0:
                x_exper_neight[j].append(x_factor_experiment[0][i]*x_factor_experiment[1][i])
            if j==1:
                x_exper_neight[j].append(x_factor_experiment[0][i]*x_factor_experiment[2][i])
            if j==2:
                x_exper_neight[j].append(x_factor_experiment[1][i]*x_factor_experiment[2][i])
            if j==3:
                x_exper_neight[j].append(x_factor_experiment[0][i]*x_factor_experiment[1][i]*x_factor_experiment[2][i])
                
    x_average_neight = []
    for i in range(len(x_exper_neight)):
        if i==0:
            x_average_neight.append(average(x_exper_neight[i], "X12_average"))
        if i==1:
            x_average_neight.append(average(x_exper_neight[i], "X13_average"))
        if i==2:
            x_average_neight.append(average(x_exper_neight[i], "X23_average"))
        if i==3:
            x_average_neight.append(average(x_exper_neight[i], "X123_average"))
            
    m = 3
    
    while True:

        y = generate_y(y_min, y_max, len(x_norm[0]), m)  # генеруємо значення функції відгуку
    
        y_average_list = []  # cереднє значення рядка Y
        for i in range(len(y[0])):
            y_average_list.append(average([y[j][i] for j in range(m)], "y_average_{}row".format(i + 1)))
    
        y_average_average = average(y_average_list, "Y_average_average")  # середнє середніх значень Y
    
        x_average = []  # cереднє стовпчика Х експеремент
        for i in range(len(x_factor_experiment)):
            x_average.append(average(x_factor_experiment[i], "X{}_average".format(i + 1)))
            
        N = len(x_norm[0])
        
        m = []
        for i in range(N):
            m.append([0 for j in range(N)])
            
        x1 = sum(x_factor_experiment[0])
        x2 = sum(x_factor_experiment[1])
        x3 = sum(x_factor_experiment[2])
        printf("x1", x1)
        printf("x2", x2)
        printf("x3", x3)
                
        m[0][0] = N
        m[0][1] = x1
        m[0][2] = x2
        m[0][3] = x3
        m[0][4] = x1*x2
        m[0][5] = x1*x3
        m[0][6] = x2*x3
        m[0][7] = x1*x2*x3
        
        m[1][0] = N*x1
        m[1][1] = x1*x1
        m[1][2] = x2*x1
        m[1][3] = x3*x1
        m[1][4] = x1*x2*x1
        m[1][5] = x1*x3*x1
        m[1][6] = x2*x3*x1
        m[1][7] = x1*x2*x3*x1
        
        m[2][0] = N*x2
        m[2][1] = x1*x2
        m[2][2] = x2*x2
        m[2][3] = x3*x2
        m[2][4] = x1*x2*x2
        m[2][5] = x1*x3*x2
        m[2][6] = x2*x3*x2
        m[2][7] = x1*x2*x3*x2
        
        m[3][0] = N*x3
        m[3][1] = x1*x3
        m[3][2] = x2*x3
        m[3][3] = x3*x3
        m[3][4] = x1*x2*x3
        m[3][5] = x1*x3*x3
        m[3][6] = x2*x3*x3
        m[3][7] = x1*x2*x3*x3
        
        m[4][0] = N*x1*x2
        m[4][1] = x1*x1*x2
        m[4][2] = x2*x1*x2
        m[4][3] = x3*x1*x2
        m[4][4] = x1*x2*x1*x2
        m[4][5] = x1*x3*x1*x2
        m[4][6] = x2*x3*x1*x2
        m[4][7] = x1*x2*x3*x1*x2        
                
        m[5][0] = N*x1*x3
        m[5][1] = x1*x1*x3
        m[5][2] = x2*x1*x3
        m[5][3] = x3*x1*x3
        m[5][4] = x1*x2*x1*x3
        m[5][5] = x1*x3*x1*x3
        m[5][6] = x2*x3*x1*x3
        m[5][7] = x1*x2*x3*x1*x3
            
        m[6][0] = N*x2*x3
        m[6][1] = x1*x2*x3
        m[6][2] = x2*x2*x3
        m[6][3] = x3*x2*x3
        m[6][4] = x1*x2*x2*x3
        m[6][5] = x1*x3*x2*x3
        m[6][6] = x2*x3*x2*x3
        m[6][7] = x1*x2*x3*x2*x3  
    
        m[7][0] = N*x1*x2*x3
        m[7][1] = x1*x1*x2*x3
        m[7][2] = x2*x1*x2*x3
        m[7][3] = x3*x1*x2*x3
        m[7][4] = x1*x2*x1*x2*x3
        m[7][5] = x1*x3*x1*x2*x3
        m[7][6] = x2*x3*x1*x2*x3
        m[7][7] = x1*x2*x3*x1*x2*x3
        
        for i in m:
            for j in i:
                print("|{: ^20}|".format(j), end="")
            print()
        
        
        
        denominator = np.array([[m[0][0],m[1][0],m[2][0],m[3][0],m[4][0],m[5][0],m[6][0],m[7][0]],
                              [m[0][1],m[1][1],m[2][1],m[3][1],m[4][1],m[5][1],m[6][1],m[7][1]],
                              [m[0][2],m[1][2],m[2][2],m[3][2],m[4][2],m[5][2],m[6][2],m[7][2]],
                              [m[0][3],m[1][3],m[2][3],m[3][3],m[4][3],m[5][3],m[6][3],m[7][3]],
                              [m[0][4],m[1][4],m[2][4],m[3][4],m[4][4],m[5][4],m[6][4],m[7][4]],
                              [m[0][5],m[1][5],m[2][5],m[3][5],m[4][5],m[5][5],m[6][5],m[7][5]],
                              [m[0][6],m[1][6],m[2][6],m[3][6],m[4][6],m[5][6],m[6][6],m[7][6]],
                              [m[0][7],m[1][7],m[2][7],m[3][7],m[4][7],m[5][7],m[6][7],m[7][7]] ])
                              
        numerator0 = np.array([[y_average_list[0],m[1][0],m[2][0],m[3][0],m[4][0],m[5][0],m[6][0],m[7][0]],
                              [y_average_list[1],m[1][1],m[2][1],m[3][1],m[4][1],m[5][1],m[6][1],m[7][1]],
                              [y_average_list[2],m[1][2],m[2][2],m[3][2],m[4][2],m[5][2],m[6][2],m[7][2]],
                              [y_average_list[3],m[1][3],m[2][3],m[3][3],m[4][3],m[5][3],m[6][3],m[7][3]],
                              [y_average_list[4],m[1][4],m[2][4],m[3][4],m[4][4],m[5][4],m[6][4],m[7][4]],
                              [y_average_list[5],m[1][5],m[2][5],m[3][5],m[4][5],m[5][5],m[6][5],m[7][5]],
                              [y_average_list[6],m[1][6],m[2][6],m[3][6],m[4][6],m[5][6],m[6][6],m[7][6]],
                              [y_average_list[7],m[1][7],m[2][7],m[3][7],m[4][7],m[5][7],m[6][7],m[7][7]] ])
                              
        numerator1 = np.array([[m[0][0],y_average_list[0],m[2][0],m[3][0],m[4][0],m[5][0],m[6][0],m[7][0]],
                              [m[0][1],y_average_list[1],m[2][1],m[3][1],m[4][1],m[5][1],m[6][1],m[7][1]],
                              [m[0][2],y_average_list[2],m[2][2],m[3][2],m[4][2],m[5][2],m[6][2],m[7][2]],
                              [m[0][3],y_average_list[3],m[2][3],m[3][3],m[4][3],m[5][3],m[6][3],m[7][3]],
                              [m[0][4],y_average_list[4],m[2][4],m[3][4],m[4][4],m[5][4],m[6][4],m[7][4]],
                              [m[0][5],y_average_list[5],m[2][5],m[3][5],m[4][5],m[5][5],m[6][5],m[7][5]],
                              [m[0][6],y_average_list[6],m[2][6],m[3][6],m[4][6],m[5][6],m[6][6],m[7][6]],
                              [m[0][7],y_average_list[7],m[2][7],m[3][7],m[4][7],m[5][7],m[6][7],m[7][7]] ])
                              
        numerator2 = np.array([[m[0][0],m[1][0],y_average_list[0],m[3][0],m[4][0],m[5][0],m[6][0],m[7][0]],
                              [m[0][1],m[1][1],y_average_list[1],m[3][1],m[4][1],m[5][1],m[6][1],m[7][1]],
                              [m[0][2],m[1][2],y_average_list[2],m[3][2],m[4][2],m[5][2],m[6][2],m[7][2]],
                              [m[0][3],m[1][3],y_average_list[3],m[3][3],m[4][3],m[5][3],m[6][3],m[7][3]],
                              [m[0][4],m[1][4],y_average_list[4],m[3][4],m[4][4],m[5][4],m[6][4],m[7][4]],
                              [m[0][5],m[1][5],y_average_list[5],m[3][5],m[4][5],m[5][5],m[6][5],m[7][5]],
                              [m[0][6],m[1][6],y_average_list[6],m[3][6],m[4][6],m[5][6],m[6][6],m[7][6]],
                              [m[0][7],m[1][7],y_average_list[7],m[3][7],m[4][7],m[5][7],m[6][7],m[7][7]] ])
                              
        numerator3 = np.array([[m[0][0],m[1][0],m[2][0],y_average_list[0],m[4][0],m[5][0],m[6][0],m[7][0]],
                              [m[0][1],m[1][1],m[2][1],y_average_list[1],m[4][1],m[5][1],m[6][1],m[7][1]],
                              [m[0][2],m[1][2],m[2][2],y_average_list[2],m[4][2],m[5][2],m[6][2],m[7][2]],
                              [m[0][3],m[1][3],m[2][3],y_average_list[3],m[4][3],m[5][3],m[6][3],m[7][3]],
                              [m[0][4],m[1][4],m[2][4],y_average_list[4],m[4][4],m[5][4],m[6][4],m[7][4]],
                              [m[0][5],m[1][5],m[2][5],y_average_list[5],m[4][5],m[5][5],m[6][5],m[7][5]],
                              [m[0][6],m[1][6],m[2][6],y_average_list[6],m[4][6],m[5][6],m[6][6],m[7][6]],
                              [m[0][7],m[1][7],m[2][7],y_average_list[7],m[4][7],m[5][7],m[6][7],m[7][7]] ])
                              
        numerator12 = np.array([[m[0][0],m[1][0],m[2][0],m[3][0],y_average_list[0],m[5][0],m[6][0],m[7][0]],
                              [m[0][1],m[1][1],m[2][1],m[3][1],y_average_list[1],m[5][1],m[6][1],m[7][1]],
                              [m[0][2],m[1][2],m[2][2],m[3][2],y_average_list[2],m[5][2],m[6][2],m[7][2]],
                              [m[0][3],m[1][3],m[2][3],m[3][3],y_average_list[3],m[5][3],m[6][3],m[7][3]],
                              [m[0][4],m[1][4],m[2][4],m[3][4],y_average_list[4],m[5][4],m[6][4],m[7][4]],
                              [m[0][5],m[1][5],m[2][5],m[3][5],y_average_list[5],m[5][5],m[6][5],m[7][5]],
                              [m[0][6],m[1][6],m[2][6],m[3][6],y_average_list[6],m[5][6],m[6][6],m[7][6]],
                              [m[0][7],m[1][7],m[2][7],m[3][7],y_average_list[7],m[5][7],m[6][7],m[7][7]] ])
                              
        numerator13 = np.array([[m[0][0],m[1][0],m[2][0],m[3][0],m[4][0],y_average_list[0],m[6][0],m[7][0]],
                              [m[0][1],m[1][1],m[2][1],m[3][1],m[4][1],y_average_list[1],m[6][1],m[7][1]],
                              [m[0][2],m[1][2],m[2][2],m[3][2],m[4][2],y_average_list[2],m[6][2],m[7][2]],
                              [m[0][3],m[1][3],m[2][3],m[3][3],m[4][3],y_average_list[3],m[6][3],m[7][3]],
                              [m[0][4],m[1][4],m[2][4],m[3][4],m[4][4],y_average_list[4],m[6][4],m[7][4]],
                              [m[0][5],m[1][5],m[2][5],m[3][5],m[4][5],y_average_list[5],m[6][5],m[7][5]],
                              [m[0][6],m[1][6],m[2][6],m[3][6],m[4][6],y_average_list[6],m[6][6],m[7][6]],
                              [m[0][7],m[1][7],m[2][7],m[3][7],m[4][7],y_average_list[7],m[6][7],m[7][7]] ])
                              
        numerator23 = np.array([[m[0][0],m[1][0],m[2][0],m[3][0],m[4][0],m[5][0],y_average_list[0],m[7][0]],
                              [m[0][1],m[1][1],m[2][1],m[3][1],m[4][1],m[5][1],y_average_list[1],m[7][1]],
                              [m[0][2],m[1][2],m[2][2],m[3][2],m[4][2],m[5][2],y_average_list[2],m[7][2]],
                              [m[0][3],m[1][3],m[2][3],m[3][3],m[4][3],m[5][3],y_average_list[3],m[7][3]],
                              [m[0][4],m[1][4],m[2][4],m[3][4],m[4][4],m[5][4],y_average_list[4],m[7][4]],
                              [m[0][5],m[1][5],m[2][5],m[3][5],m[4][5],m[5][5],y_average_list[5],m[7][5]],
                              [m[0][6],m[1][6],m[2][6],m[3][6],m[4][6],m[5][6],y_average_list[6],m[7][6]],
                              [m[0][7],m[1][7],m[2][7],m[3][7],m[4][7],m[5][7],y_average_list[7],m[7][7]] ])
                              
        numerator123 = np.array([[m[0][0],m[1][0],m[2][0],m[3][0],m[4][0],m[5][0],m[6][0],y_average_list[0]],
                              [m[0][1],m[1][1],m[2][1],m[3][1],m[4][1],m[5][1],m[6][1],y_average_list[1]],
                              [m[0][2],m[1][2],m[2][2],m[3][2],m[4][2],m[5][2],m[6][2],y_average_list[2]],
                              [m[0][3],m[1][3],m[2][3],m[3][3],m[4][3],m[5][3],m[6][3],y_average_list[3]],
                              [m[0][4],m[1][4],m[2][4],m[3][4],m[4][4],m[5][4],m[6][4],y_average_list[4]],
                              [m[0][5],m[1][5],m[2][5],m[3][5],m[4][5],m[5][5],m[6][5],y_average_list[5]],
                              [m[0][6],m[1][6],m[2][6],m[3][6],m[4][6],m[5][6],m[6][6],y_average_list[6]],
                              [m[0][7],m[1][7],m[2][7],m[3][7],m[4][7],m[5][7],m[6][7],y_average_list[7]] ])
                              
        try:
            b0 = np.linalg.det(numerator0) / np.linalg.det(denominator)
            b1 = np.linalg.det(numerator1) / np.linalg.det(denominator)
            b2 = np.linalg.det(numerator2) / np.linalg.det(denominator)
            b3 = np.linalg.det(numerator3) / np.linalg.det(denominator)
            b12 = np.linalg.det(numerator12) / np.linalg.det(denominator)
            b23 = np.linalg.det(numerator23) / np.linalg.det(denominator)
            b13 = np.linalg.det(numerator13) / np.linalg.det(denominator)
            b123 = np.linalg.det(numerator123) / np.linalg.det(denominator)
            
            break
            
            print("y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1*x2 + {}*x1*x3 + {}*x2*x3 + {}*x1*x2*x3".format(str(b0), str(b1), str(b2), str(b3), str(b12), str(b13), str(b23), str(b123)))
        except:
            print("Неможливо побудувати рівняння регресії")
            m+=1
        
    
else:
    print("За критерієм Фішера рівняння регресії адекватно оригіналу при рівні значимості 0.05")
    Krit_Fish = "Адекв."
    


    """Таблиця з головными результатами розрахунків"""
    print("\nТаблиця результату:")
    
    for j in range(4):
        print("|{: ^9}|".format("x" + str(j) + "factor"), end="")
    for j in range(3):
        print("|{: ^9}|".format("x" + str(j + 1)), end="")
    for j in range(m):
        print("|{: ^9}|".format("y" + str(j + 1)), end="")
    print("|{: ^9}||{: ^9}||{: ^9}||{: ^9}||{: ^9}|"
          .format("Y_mid", "Y_mid_exp", "Стьюдент", "Krit_Kohr", "Krit_Fish"))
    print("{:-^165}".format("-"))
    for i in range(4):
        for j in range(4):
            print("|{: ^9}|".format(x_norm[j][i]), end="")
        for j in range(3):
            print("|{: ^9}|".format(x_factor_experiment[j][i]), end="")
        for j in range(m):
            print("|{: ^9}|".format(y[j][i]), end="")
        print("|{: ^9}||{: ^9}||{: ^9}|".format(r(y_average_list[i]), r(y_average_row_controls[i]),
                                                   r(y_average_row_Student[i])), end="")
        if i == 0:
            print("|{: ^9}||{: ^9}|".format(Krit_Kohr, Krit_Fish), end="")
        print()
