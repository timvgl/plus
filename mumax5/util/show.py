import matplotlib.pyplot as plt



def show_layer(quantity, component=0, layer=0):
    f = quantity.eval()
    plt.title(quantity.name)
    plt.imshow(f[component, layer])
    plt.show()
