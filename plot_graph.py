from matplotlib import pyplot as plt
import numpy as np

def get_file(path):
    estposes = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            line_list =[float(x) for x in line]
            estposes.append(line_list)
    estposes_np = np.array(estposes)
    return estposes_np

def plot_traj(estposes, vis=False, savefigname='results/android_1.png', title=''):
    fig = plt.figure(figsize=(4,4))
    cm = plt.cm.get_cmap('Spectral')

    plt.subplot(111)
    #plt.plot(gtposes[:,0],gtposes[:,1], linestyle='dashed',c='k')
    #plt.plot(gtposes[:,0],gtposes[:,2], linestyle='dashed',c='k')
    plt.plot(estposes[:,0],estposes[:,1],c='#ff7f0e')
    #plt.plot(estposes[:, 0], estposes[:, 1],c='#ff7f0e')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend(['TartanVO'])
    plt.title(title)
    if savefigname is not None or True:
        print('save figure to {}'.format(savefigname))
        plt.savefig(savefigname)
    if vis:
        plt.show()
    plt.close(fig)


def main():
    estposes = get_file('results/android_tartanvo_1914.txt')
    print("len(estposes): ", len(estposes))
    plot_traj(estposes, vis=False, title='')

if __name__ == '__main__':
    main()