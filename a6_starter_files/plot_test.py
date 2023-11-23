"""plot_test.py

This can be used to determine whether or not
matplotlib has been successfully installed.
"""

try:
    import matplotlib.pyplot as plt
    try:
        import random
        r = lambda: random.random()*10
        xdata = [r() for i in range(10)]
        ydata = [r() for i in range(10)]
        print("data: ", xdata, ydata)
        plt.plot(xdata, ydata, marker='o', linestyle='')
        plt.show()
        print("Your matplotlib installation seems fine.")
    except:
        pass
except:
    print("Could not import matplotlib")
    print("see https://www.w3schools.com/python/matplotlib_getting_started.asp")


    
        
