import argparse
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt 
import scipy as sp
import numpy as np
import matplotlib.animation as animation


class Pendulum:
    """ A system of two pendulums moving based on the initial conditions """
    def __init__(self, theta1, theta2, dt=0.01):
        # the initial angles of the rods
        self.theta1 = theta1
        self.theta2 = theta2

        # the momenta of the two pendulums
        self.p1 = 0.0
        self.p2 = 0.0

        # the change in time
        self.dt = dt

        # the acceleration due to gravity on Earth
        self.g = 9.81
        # set the default length of the rods of the two pendulums to 1.0 (identical)
        # the same thing happens to the masses of the two pendulums
        self.length = 1.0

        # trajectory implemented by Cartesian coordinates
        self.trajectory = [self.polar_to_cartesian()]
  
    def polar_to_cartesian(self):
        """ Convert the Polar Coordinates to the equivalent Cartesian Coordinates """
        x1 =  self.length * np.sin(self.theta1)        
        y1 = -self.length * np.cos(self.theta1)
          
        x2 = x1 + self.length * np.sin(self.theta2)
        y2 = y1 - self.length * np.cos(self.theta2)

        return np.array([[0.0, 0.0], [x1, y1], [x2, y2]])
      
    def evolve(self):
        """ Move the pendulums to a new state/position """
        theta1 = self.theta1
        theta2 = self.theta2
        p1 = self.p1
        p2 = self.p2
        g = self.g
        l = self.length

        # Calculate using Hamilton's equations of motions
        expr1 = np.cos(theta1 - theta2)
        expr2 = np.sin(theta1 - theta2)
        expr3 = (1 + expr2**2)
        expr4 = p1 * p2 * expr2 / expr3
        expr5 = (p1**2 + 2 * p2**2 - p1 * p2 * expr1) \
        * np.sin(2 * (theta1 - theta2)) / 2 / expr3**2
        expr6 = expr4 - expr5
         
        self.theta1 += self.dt * (p1 - p2 * expr1) / expr3
        self.theta2 += self.dt * (2 * p2 - p1 * expr1) / expr3
        self.p1 += self.dt * (-2 * g * l * np.sin(theta1) - expr6)
        self.p2 += self.dt * (    -g * l * np.sin(theta2) + expr6)
         
        new_position = self.polar_to_cartesian()
        # Put the new position to the trajectory of the second pendulum
        self.trajectory.append(new_position)
        return new_position
 
 
class Animator:
    """ A class used to illustrate the pendulum system """
    def __init__(self, pendulum, draw_trace=False):
        self.pendulum = pendulum
        self.draw_trace = draw_trace
        self.time = 0.0
  
        # set up the figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.set_xlim(-2.5, 2.5)
  
        # prepare a text window for the timer
        self.time_text = self.ax.text(0.05, 0.95, '', 
            horizontalalignment='left', 
            verticalalignment='top', 
            transform=self.ax.transAxes)
  
        # initialize by plotting the last position of the trajectory
        self.line, = self.ax.plot(
            self.pendulum.trajectory[-1][:, 0], 
            self.pendulum.trajectory[-1][:, 1], 
            'bo--', lw=3, markersize=8)
          
        # trace the whole trajectory of the second pendulum mass
        if self.draw_trace:
            self.trace, = self.ax.plot(
                [a[2, 0] for a in self.pendulum.trajectory],
                [a[2, 1] for a in self.pendulum.trajectory],
                'g', lw=2, markersize=1)
     
    def advance_time_step(self):
        """ continuously change the time by dt and move the pendulums """
        while True:
            self.time += self.pendulum.dt
            yield self.pendulum.evolve()
             
    def update(self, data):
        """ Update the current state of the pendulums """
        self.time_text.set_text('Elapsed time: {:6.2f} s'.format(self.time))
         
        self.line.set_ydata(data[:, 1])
        self.line.set_xdata(data[:, 0])

        # if the tracing option is set, create the trajectory
        if self.draw_trace:
            self.trace.set_xdata([a[2, 0] for a in self.pendulum.trajectory])
            self.trace.set_ydata([a[2, 1] for a in self.pendulum.trajectory])
        return self.line,
     
    def animate(self):
        """ Animate the pendulum system """
        self.animation = animation.FuncAnimation(self.fig, self.update,
                         self.advance_time_step, interval=25, blit=False)
        
    def save(self, output_name): 
        self.animation.save(f'{output_name}.gif',writer='pillow',fps=25)

        
def generate_file_name():
    # Helper function used when saving the simulation to a file
    x = datetime.now()
    return x.strftime('%b %d %Y %H-%M-%S')


def main():
    """ The driver class to simulate the two pendulum systems """

    DEFAULT_ANGLE = sp.pi

    # Argument parsing... allows user to specify the length, mass, and angle
    # of each pendulum
    parser = argparse.ArgumentParser(
        description='Visualize a double pendulum simulation.'
    )

    parser.add_argument('-p1_a', '--pen1a_angle', type=float,
                        default=DEFAULT_ANGLE,
                        help='starting angle of pendulum 1 (degrees)')

    parser.add_argument('-p2_a', '--pen2a_angle', type=float,
                        default=DEFAULT_ANGLE-0.01,
                        help='starting angle of pendulum 2 (degrees)')
    
    parser.add_argument('-p1_b', '--pen1b_angle', type=float,
                        default=DEFAULT_ANGLE,
                        help='starting angle of pendulum 1 (degrees)')

    parser.add_argument('-p2_b', '--pen2b_angle', type=float,
                        default=DEFAULT_ANGLE-0.03,
                        help='starting angle of pendulum 2 (degrees)')

    parser.add_argument('-o1', '--output_file_1',
                        default='sys1-'+generate_file_name(),
                        help='name of file to be saved for the first pendulum system')
    
    parser.add_argument('-o2', '--output_file_2',
                        default='sys2-'+generate_file_name(),
                        help='name of file to be saved for the second pendulum system')

    args = parser.parse_args()
    print(args)
    
    pendulum1 = Pendulum(theta1=args.pen1a_angle, theta2=args.pen2a_angle, dt=0.01)
    animator1 = Animator(pendulum=pendulum1, draw_trace=True)

    pendulum2 = Pendulum(theta1=args.pen1b_angle, theta2=args.pen2b_angle, dt=0.01)
    animator2 = Animator(pendulum=pendulum2, draw_trace=True)

    animator1.animate()
    animator2.animate()

    animator1.save(args.output_file_1)
    animator2.save(args.output_file_2)

    plt.show()


if __name__ == '__main__':
    main()
