def plot_save(write_png, folder, filename):
    import matplotlib.pyplot as plt
    if write_png:
        if folder is not None:
            png = folder + filename + '.png'
            print('Saving', png)
            plt.savefig(png, dpi=100)
        else:
            print('Ha puesto write_png pero no ha especificado el folder')


def radar_plot(alice, bob, concepts, desc=None, write_png=True, folder=None, filename=''):
    """
subjects = ['PHY', 'CHEM', 'BIO', 'MATH', 'ECO']
alice = [60, 40, 68, 94, 27]
bob = [81, 30, 75, 37, 46]
    Parameters
    ----------
    alice
    bob
    concepts
    desc
    """
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    angles = np.linspace(0, 2 * np.pi, len(concepts), endpoint=False)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    # Alice Plot
    ax.plot(angles, alice, 'o-', color='g', linewidth=1, label='Cluster')
    ax.fill(angles, alice, alpha=0.25, color='g')
    # Bob Plot
    ax.plot(angles, bob, 'o-', color='orange', linewidth=1, label='Other')
    ax.fill(angles, bob, alpha=0.25, color='orange')

    if desc is None:
        labels = concepts
    else:
        labels = [desc[x] for x in concepts]
    ax.set_thetagrids(angles * 180 / np.pi, labels)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plot_save(write_png, folder, filename)
    plt.show()


def plot_hist(lista, bins=10):
    import matplotlib.pyplot as plt
    # create histogram from list of data
    plt.hist(lista, bins=bins)
