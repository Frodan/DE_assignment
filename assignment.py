import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import CheckButtons, RadioButtons
from matplotlib.widgets import TextBox


class Solution:
    def __init__(self, ax_sol, ax_err):
        self.x0 = round(np.pi, 2)
        self.y0 = 1
        self.x1 = round(np.pi * 4, 2)
        self.n = 0.1
        self.ax_sol = ax_sol
        self.ax_err = ax_err

        self.n_start = self.n
        self.n_end = 1
        self.nn = 0.1

    def plot(self):
        pass

    def update(self, x0=None, x1=None, y0=None, n=None, n_start=None, n_end=None, nn=None):
        if x0:
            self.x0 = x0
        if x1:
            self.x1 = x1
        if y0:
            self.y0 = y0
        if n:
            self.n = n
        if n_start:
            self.n_start = n_start
        if n_end:
            self.n_end = n_end
        if nn:
            self.nn = nn

    def get_xlist(self):
        return np.arange(self.x0, self.x1, self.n)

    def get_ylist(self, xlist):
        pass

    def get_error_xlist(self):
        return np.arange(self.n_start, self.n_end, self.nn)

    def get_global_error_list(self, exact_ylist):
        xlist = self.get_xlist()
        ylist = self.get_ylist(xlist)
        return [abs(ylist[i] - exact_ylist[i]) for i in range(len(exact_ylist))]

    def get_local_error_list(self, exact_ylist):
        local_error_ylist = self.get_global_error_list(exact_ylist)

        for i in range(1, len(local_error_ylist)):
            local_error_ylist[i] = abs(local_error_ylist[i] - local_error_ylist[i - 1])
        return local_error_ylist

    def get_global_error_ylist(self):
        pass

    def get_global_error_plot(self):
        pass

    def help_func(self):
        pass


class ExactMethod(Solution):
    def plot(self):
        xlist = self.get_xlist()
        ylist = self.get_ylist(xlist)
        return self.ax_sol.plot(xlist, ylist, label='Exact', color='blue')

    def const(self):
        return self.y0/self.x0 - np.sin(self.x0)

    def get_ylist(self, xlist):
        const = self.const()
        return [x * np.sin(x) + x * const for x in xlist]


class EulerMethod(Solution):
    def plot(self):
        xlist = self.get_xlist()
        ylist = self.get_ylist(xlist)
        return self.ax_sol.plot(xlist, ylist, label='Euler', color='green')

    def local_error_plot(self, exact_ylist):
        xlist = self.get_xlist()
        ylist = self.get_local_error_list(exact_ylist)
        return self.ax_err.plot(xlist, ylist, label='Euler', color='green')

    def get_ylist(self, xlist):
        y = self.y0
        ylist = [y]
        if len(xlist > 1):
            for i in range(1, len(xlist)):
                y = ylist[i-1] + self.n * self.help_func(xlist[i-1], ylist[i-1])
                ylist.append(y)
        return ylist

    def help_func(self, x, y):
        return y/x + x * np.cos(x)

    def get_global_error_ylist(self):
        exact = ExactMethod(self.ax_sol, self.ax_err)
        euler = EulerMethod(self.ax_sol, self.ax_err)

        error_ylist = []
        error_xlist = self.get_error_xlist()
        for i in error_xlist:
            exact.update(x0=self.x0, x1=self.x1, n=i)
            exact_xlist = exact.get_xlist()
            exact_ylist = exact.get_ylist(exact_xlist)
            euler.update(x0=self.x0, x1=self.x1, n=i)
            euler_ylist = euler.get_ylist(exact_xlist)
            diff = [abs(exact_ylist[i] - euler_ylist[i]) for i in range(len(exact_xlist))]
            max_error = max(diff)
            error_ylist.append(max_error)

        return error_ylist

    def get_global_error_plot(self):
        xlist = self.get_error_xlist()
        ylist = self.get_global_error_ylist()
        print(self, xlist, ylist)
        return self.ax_err.plot(xlist, ylist, label='Euler', color='green')


class ImprovedEulerMethod(Solution):
    def plot(self):
        xlist = self.get_xlist()
        ylist = self.get_ylist(xlist)
        return self.ax_sol.plot(xlist, ylist, label='Imp. Euler', color='red')

    def local_error_plot(self, exact_ylist):
        xlist = self.get_xlist()
        ylist = self.get_local_error_list(exact_ylist)
        return self.ax_err.plot(xlist, ylist, label='Imp. Euler', color='red')

    def get_ylist(self, xlist):
        y = self.y0
        ylist = [y]
        if len(xlist > 1):
            for i in range(1, len(xlist)):
                arg1 = xlist[i - 1] + self.n / 2
                arg2 = ylist[i - 1] + self.n / 2 * self.help_func(xlist[i - 1], ylist[i - 1])
                y = ylist[i - 1] + self.n * self.help_func(arg1, arg2)
                ylist.append(y)
        return ylist

    def help_func(self, x, y):
        return y/x + x * np.cos(x)

    def get_global_error_ylist(self):
        exact = ExactMethod(self.ax_sol, self.ax_err)
        impeuler = ImprovedEulerMethod(self.ax_sol, self.ax_err)

        error_ylist = []
        error_xlist = self.get_error_xlist()
        for i in error_xlist:
            exact.update(x0=self.x0, x1=self.x1, n=i)
            exact_xlist = exact.get_xlist()
            exact_ylist = exact.get_ylist(exact_xlist)
            impeuler.update(x0=self.x0, x1=self.x1, n=i)
            impeuler_ylist = impeuler.get_ylist(exact_xlist)
            diff = [abs(exact_ylist[i] - impeuler_ylist[i]) for i in range(len(exact_xlist))]
            max_error = max(diff)
            error_ylist.append(max_error)

        return error_ylist

    def get_global_error_plot(self):
        xlist = self.get_error_xlist()
        ylist = self.get_global_error_ylist()
        return self.ax_err.plot(xlist, ylist, label='Imp. Euler', color='red')


class RungeKuttaMethod(Solution):
    def plot(self):
        xlist = self.get_xlist()
        ylist = self.get_ylist(xlist)
        return self.ax_sol.plot(xlist, ylist, label='Runge-Kutta Method', color='yellow')

    def local_error_plot(self, exact_ylist):
        xlist = self.get_xlist()
        ylist = self.get_local_error_list(exact_ylist)
        return self.ax_err.plot(xlist, ylist, label='Runge-Kutta Method', color='yellow')

    def get_ylist(self, xlist):
        y = self.y0
        ylist = [y]
        if len(xlist > 1):
            for i in range(1, len(xlist)):
                k1 = self.help_func(xlist[i - 1], ylist[i - 1])
                k2 = self.help_func(xlist[i - 1] + self.n / 2, ylist[i - 1] + self.n / 2 * k1)
                k3 = self.help_func(xlist[i - 1] + self.n / 2, ylist[i - 1] + self.n / 2 * k2)
                k4 = self.help_func(xlist[i - 1] + self.n, ylist[i - 1] + self.n * k3)
                y = ylist[i - 1] + self.n / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                ylist.append(y)
        return ylist

    def help_func(self, x, y):
        return y/x + x * np.cos(x)

    def get_global_error_ylist(self):
        exact = ExactMethod(self.ax_sol, self.ax_err)
        runge_kutta = RungeKuttaMethod(self.ax_sol, self.ax_err)

        error_ylist = []
        error_xlist = self.get_error_xlist()
        for i in error_xlist:
            exact.update(x0=self.x0, x1=self.x1, n=i)
            exact_xlist = exact.get_xlist()
            exact_ylist = exact.get_ylist(exact_xlist)
            runge_kutta.update(x0=self.x0, x1=self.x1, n=i)
            runge_kutta_ylist = runge_kutta.get_ylist(exact_xlist)
            diff = [abs(exact_ylist[i] - runge_kutta_ylist[i]) for i in range(len(exact_xlist))]
            max_error = max(diff)
            error_ylist.append(max_error)

        return error_ylist

    def get_global_error_plot(self):
        xlist = self.get_error_xlist()
        ylist = self.get_global_error_ylist()
        return self.ax_err.plot(xlist, ylist, label='Runge-Kutta Method', color='yellow')


class Grid:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        plt.subplots_adjust(left=0.2)

        self.ax.set_title("Solutions graph")
        self.ax2.set_title("Error graph")

        self.exact_method = ExactMethod(self.ax, self.ax2)
        self.euler_method = EulerMethod(self.ax, self.ax2)
        self.imp_euler_method = ImprovedEulerMethod(self.ax, self.ax2)
        self.runge_kutta_method = RungeKuttaMethod(self.ax, self.ax2)

        self.methods = self.get_methods_list()
        self.plots = self.get_plots_list()
        self.local_error_plots = self.get_local_error_plots_list()
        self.global_error_plots = []
        self.error_flag = True

        rax = plt.axes([0.03, 0.75, 0.13, 0.13])
        self.labels = [str(line.get_label()) for line in self.plots]
        visibility = [line.get_visible() for line in self.plots]
        self.check = CheckButtons(rax, self.labels, visibility)
        self.check.on_clicked(self.update_plot_by_checkbuttons)

        error_rax = plt.axes([0.03, 0.35, 0.13, 0.1])
        self.radio = RadioButtons(error_rax, ("Show local error", "Show global error"))
        self.radio.on_clicked(self.change_error_plots)

        x0box = plt.axes([0.1, 0.71, 0.03, 0.03])
        x1box = plt.axes([0.1, 0.68, 0.03, 0.03])
        y0box = plt.axes([0.1, 0.65, 0.03, 0.03])
        nbox = plt.axes([0.1, 0.62, 0.03, 0.03])

        self.x0_box = TextBox(x0box, 'x0:', initial="3.14")
        self.x1_box = TextBox(x1box, 'x1:', initial="12.56")
        self.y0_box = TextBox(y0box, 'y0:', initial="1")
        self.n_box = TextBox(nbox, 'n:', initial="0.1")

        nstartbox = plt.axes([0.1, 0.31, 0.03, 0.03])
        nendbox = plt.axes([0.1, 0.28, 0.03, 0.03])
        nnbox = plt.axes([0.1, 0.25, 0.03, 0.03])
        self.n_start = TextBox(nstartbox, 'Start n:', initial="0.1")
        self.n_end = TextBox(nendbox, 'End n:', initial="1")
        self.n_n = TextBox(nnbox, 'Step:', initial="0.1")

        self.x0_box.on_submit(self.submit_x0)
        self.x1_box.on_submit(self.submit_x1)
        self.y0_box.on_submit(self.submit_y0)
        self.n_box.on_submit(self.submit_n)

        self.n_start.on_submit(self.submit_n_start)
        self.n_end.on_submit(self.submit_n_end)
        self.n_n.on_submit(self.submit_nn)

        self.ax.legend(loc='lower left')
        self.ax.set_ylabel("y")
        self.ax.set_xlabel("x")
        self.ax2.set_ylabel('y')
        self.ax2.set_xlabel("x")

    def get_methods_list(self):
        return [self.exact_method, self.euler_method, self.imp_euler_method, self.runge_kutta_method]

    def get_plots_list(self):
        return [method.plot()[0] for method in self.get_methods_list()]

    def get_local_error_plots_list(self):
        exact_ylist = self.exact_method.get_ylist(self.exact_method.get_xlist())
        return [x.local_error_plot(exact_ylist)[0] for x in self.methods[1:]]

    def get_global_error_plots_list(self):
        return [x.get_global_error_plot()[0] for x in self.methods[1:]]

    def update_plot_by_checkbuttons(self, label):
        index = self.labels.index(label)
        self.plots[index].set_visible(not self.plots[index].get_visible())
        plt.draw()

    def change_error_plots(self, label):
        if label == 'Show local error':
            self.local_error_plots = self.get_local_error_plots_list()
            for plot in self.global_error_plots:
                plot.remove()
            self.error_flag = True
            self.ax2.set_ylabel('y')
            self.ax2.set_xlabel("x")
        else:
            self.global_error_plots = self.get_global_error_plots_list()
            for plot in self.local_error_plots:
                plot.remove()
            self.error_flag = False
            self.ax2.set_ylabel('error')
            self.ax2.set_xlabel("step")

        self.ax2.relim()
        self.ax2.autoscale_view()
        plt.draw()

    def submit_x0(self, data):
        x0 = float(data)
        self.update_plots(x0=x0)

    def submit_x1(self, data):
        x1 = float(data)
        self.update_plots(x1=x1)

    def submit_y0(self, data):
        y0 = float(data)
        self.update_plots(y0=y0)

    def submit_n(self, data):
        n = float(data)
        self.update_plots(n=n)

    def submit_n_start(self, data):
        n_start = float(data)
        self.update_plots(n_start=n_start)

    def submit_n_end(self, data):
        n_end = float(data)
        self.update_plots(n_end=n_end)

    def submit_nn(self, data):
        nn = float(data)
        self.update_plots(n_start=nn)

    def update_plots(self, x0=None, x1=None, y0=None, n=None, n_start=None, n_end=None, nn=None):
        for i, x in enumerate(self.methods):
            x.update(x0, x1, y0, n, n_start, n_end, nn)

            new_xlist = x.get_xlist()
            new_ylist = x.get_ylist(new_xlist)

            self.plots[i].set_xdata(new_xlist)
            self.plots[i].set_ydata(new_ylist)

            if i > 0:
                exact_xlist = self.exact_method.get_xlist()
                exact_ylist = self.exact_method.get_ylist(exact_xlist)

                if self.error_flag:
                    local_error_ylist = x.get_local_error_list(exact_ylist)
                    self.local_error_plots[i - 1].set_xdata(new_xlist)
                    self.local_error_plots[i - 1].set_ydata(local_error_ylist)
                else:
                    global_error_xlist = x.get_error_xlist()
                    global_error_ylist = x.get_global_error_ylist()
                    self.global_error_plots[i - 1].set_xdata(global_error_xlist)
                    self.global_error_plots[i - 1].set_ydata(global_error_ylist)

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        plt.draw()

    def show(self):
        plt.show()


if __name__ == "__main__":
    grid = Grid()
    grid.show()
