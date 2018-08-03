import matplotlib.pyplot as plt
import numpy as np
import sys

class ClickAndRoi:
    def __init__(self, fig = [], ax = []):
        
        self.selected_pixel = []              

                     
        self.fig = fig
        self.ax = ax
        self._ID1 = self.fig.canvas.mpl_connect('button_press_event', self._onclick)
        plt.waitforbuttonpress()
        fig.canvas.mpl_disconnect(self._ID1)

#        if sys.flags.interactive:
#            plt.show(block=False)
#        else:
#            plt.show()
        if sys.flags.interactive:
            pass
        else:        
            plt.close(self.fig)
        pass              
              
        
    def _onclick(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.selected_pixel = [x, y]
            self.ax.scatter(self.selected_pixel[0],
                            self.selected_pixel[1],
                            s=5,
                            c='red',
                            marker='o')
            plt.draw()
            self.fig.canvas.mpl_disconnect(self._ID1)

            if sys.flags.interactive:
                pass
            else:        
                plt.close(self.fig)
            pass              
                  

#
