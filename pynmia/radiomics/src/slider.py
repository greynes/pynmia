class plotter:
    def __init__(self, im, i=0):    
        import matplotlib.pyplot as plt            
        self.im = im
        self.i = i
        self.vmin = im.min()
        self.vmax = im.max()
        self.fig = plt.figure()
        plt.gray()
        self.ax = self.fig.add_subplot(111)
        self.draw()
        self.fig.canvas.mpl_connect('key_press_event',self)

    def draw(self):
        import matplotlib.pyplot as plt            
        if self.im.ndim is 2:
            im = self.im
        if self.im.ndim is 3:
            im = self.im[...,self.i]
            self.ax.set_title('image {0}'.format(self.i))

        plt.show()

        self.ax.imshow(im, vmin=self.vmin, vmax=self.vmax, interpolation=None)


    def __call__(self, event):
        old_i = self.i
        if event.key=='right':
            self.i = min(self.im.shape[2]-1, self.i+1)
        elif event.key == 'left':
            self.i = max(0, self.i-1)
        if old_i != self.i:
            self.draw()
            self.fig.canvas.draw()


def slice_show(im, i=0):
    plotter(im, i)