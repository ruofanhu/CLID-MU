## https://github.com/pxiangwu/TopoFilter/blob/master/noise.py

import numpy as np
from scipy import stats
from math import inf
import torch.nn.functional as F
import torch
import os
def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = noise / (size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (1 - noise) * np.ones(size))
    return P


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def flip_labels_C(y_train, noise, nb_classes,random_state=None):
    '''
    returns a matrix with (1 - noise) on the diagonals, and noise
    concentrated in only one other entry for each row
    '''
    
    C = np.eye(nb_classes) * (1 - noise)
    row_indices = np.arange(nb_classes)
    for i in range(nb_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = noise
        
    y_train_noisy = multiclass_noisify(y_train, P=C,
                                       random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    keep_indices = np.where(y_train_noisy == y_train)[0]
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)
    y_train = y_train_noisy

    return y_train, C, keep_indices    
    


def flip_labels_C_two(y_train, noise, nb_classes,random_state=None):
    '''
    returns a matrix with (1 - noise) on the diagonals, and noise
    concentrated in only one other entry for each row
    '''

    C = np.eye(nb_classes) * (1 - noise)
    row_indices = np.arange(nb_classes)
    for i in range(nb_classes):
        C[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = noise / 2
    y_train_noisy = multiclass_noisify(y_train, P=C,
                                       random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    keep_indices = np.where(y_train_noisy == y_train)[0]
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)
    y_train = y_train_noisy

    return y_train, C, keep_indices     

def noisify_pairflip(y_train, nb_classes, noise, random_state=None):
    #  Pairflip noise flips noisy labels into their adjacent class. 
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n
    
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train, P, keep_indices


def noisify_with_P(y_train, nb_classes, noise, random_state=None):

    if noise > 0.0:
        P = build_uniform_P(nb_classes, noise)
        # seed the random numbers with #run
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]
        if actual_noise==0:
            y_train_noisy[-1]=1-y_train_noisy[-1]
            actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        c_n={}
        for i in range(nb_classes):
            ids=np.where(y_train == i)[0]
            n_noise= np.sum(y_train_noisy[ids]!=i)/len(ids)
            c_n[i]=n_noise
        print('Actual noise %.2f' % actual_noise, c_n)
    
        y_train = y_train_noisy
            
    else:
        P = np.eye(nb_classes)
        keep_indices = np.arange(len(y_train))

    return y_train, P, keep_indices


def noisify_cifar10_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        automobile <- truck
        bird -> airplane
        cat <-> dog
        deer -> horse
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    keep_indices = np.arange(len(y_train))
    n = noise

    if n > 0.0:
        # automobile <- truck
        P[9, 9], P[9, 1] = 1. - n, n

        # bird -> airplane
        P[2, 2], P[2, 0] = 1. - n, n

        # cat <-> dog
        P[3, 3], P[3, 5] = 1. - n, n
        P[5, 5], P[5, 3] = 1. - n, n

        # automobile -> truck
        P[4, 4], P[4, 7] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P, keep_indices


def noisify_modelnet40_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        automobile <- truck
        bird -> airplane
        cat <-> dog
        deer -> horse
    """
    nb_classes = 40
    P = np.eye(nb_classes)
    keep_indices = np.arange(len(y_train))
    n = noise

    if n > 0.0:
        # bench -> chair
        P[3, 3], P[3, 9] = 1. - n, n

        # bottle <-> vase
        P[5, 5], P[5, 37] = 1. - n, n
        P[37, 37], P[37, 5] = 1. - n, n

        # desk <-> table
        P[12, 12], P[12, 33] = 1. - n, n
        P[33, 33], P[33, 12] = 1. - n, n

        # flower_pot <-> glass box
        P[15, 15], P[15, 16] = 1. - n, n
        P[16, 16], P[16, 15] = 1. - n, n

        # bowel <-> cup
        P[6, 6], P[6, 10] = 1. - n, n
        P[10, 10], P[10, 6] = 1. - n, n

        # night stand -> table
        P[23, 23], P[23, 33] = 1. - n, n

        # tv stand -> table
        P[36, 36], P[36, 33] = 1. - n, n

        # sofa -> bench
        P[30, 30], P[30, 3] = 1. - n, n

        # bathhub -> sink
        P[1, 1], P[1, 29] = 1. - n, n

        # dresser <-> wardrobe
        P[14, 14], P[14, 38] = 1. - n, n
        P[38, 38], P[38, 14] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P, keep_indices


def build_for_cifar100(size, noise):
    """ The noise matrix flips to the "next" class with probability 'noise'.
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise

    # adjust last row
    P[size-1, 0] = noise

    return P


def noisify_cifar100_asymmetric(y_train, noise, random_state=None):
    """mistakes are inside the same superclass of 10 classes, e.g. 'fish'
    """
    nb_classes = 100
    P = np.eye(nb_classes)
    n = noise
    nb_superclasses = 20
    nb_subclasses = 5

    keep_indices = np.arange(len(y_train))

    if n > 0.0:
        for i in np.arange(nb_superclasses):
            init, end = i * nb_subclasses, (i+1) * nb_subclasses
            P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P, keep_indices


def noisify_mnist_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        1 <- 7
        2 -> 7
        3 -> 8
        5 <-> 6
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise

    keep_indices = np.arange(len(y_train))

    if n > 0.0:
        # 1 <- 7
        P[7, 7], P[7, 1] = 1. - n, n

        # 2 -> 7
        P[2, 2], P[2, 7] = 1. - n, n

        # 5 <-> 6
        P[5, 5], P[5, 6] = 1. - n, n
        P[6, 6], P[6, 5] = 1. - n, n

        # 3 -> 8
        P[3, 3], P[3, 8] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise,n)
        
        y_train = y_train_noisy

    return y_train, P, keep_indices


def noisify_binary_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        1 -> 0: n
        0 -> 1: .05
    """
    P = np.eye(2)
    n = noise

    keep_indices = np.arange(len(y_train))

    assert 0.0 <= n < 0.5

    if noise > 0.0:
        P[1, 1], P[1, 0] = 1.0 - n, n
        P[0, 0], P[0, 1] = 0.95, 0.05

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P, keep_indices




def get_instance_noisy_label(n, newdataset, labels, num_classes, feature_size, norm_std, seed):
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    if torch.cuda.is_available():
        labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)
    if torch.cuda.is_available():
        W = torch.FloatTensor(W).cuda()
    else:
        W = torch.FloatTensor(W)
    for i, (x, y) in enumerate(newdataset):
        if torch.cuda.is_available():
            x = x.cuda()
            x = x.reshape(feature_size)

        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l1 = [i for i in range(label_num)]
    new_label = [np.random.choice(l1, p=P[i]) for i in range(labels.shape[0])]

    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1

    return np.array(new_label)

def noisify_instance(train_data,train_labels,num_class,noise_rate,random_state):

    np.random.seed(random_state)

    q_ = np.random.normal(loc=noise_rate,scale=0.1,size=1000000)
    q = []
    for pro in q_:
        if 0 < pro < 1:
            q.append(pro)
        if len(q)==50000:
            break

    w = np.random.normal(loc=0,scale=1,size=(32*32*3,num_class))

    noisy_labels = []
    for i, sample in enumerate(train_data):
        sample = sample.flatten()
        p_all = np.matmul(sample,w)
        p_all[train_labels[i]] = -1000000
        p_all = q[i]* F.softmax(torch.tensor(p_all),dim=0).numpy()
        p_all[train_labels[i]] = 1 - q[i]
        noisy_labels.append(np.random.choice(np.arange(num_class),p=p_all/sum(p_all)))
    over_all_noise_rate = 1 - float(torch.tensor(train_labels).eq(torch.tensor(noisy_labels)).sum())/50000
    print('Actual noise %.2f' % over_all_noise_rate)
    return np.array(noisy_labels)


def get_noisy_label(dataset,y_train,num_classes,noise_type,noise_ratio,train_data,data_dir,seed):

    if noise_type=='sym':
        
        noise_y_train, p, keep_indices = noisify_with_P(y_train, nb_classes=num_classes, noise=noise_ratio, random_state=seed)
        if len(keep_indices)==0:
            ids=random.sample(range(len(y_train)), round(len(y_train)*noise_ratio)) 
            noise_y_train = np.array(y_train)
            for i in ids:
                noise_y_train[i]=random.randint(0,num_classes)
    elif noise_type=='asy':
        if dataset == 'cifar10':
            noise_y_train, p, keep_indices = noisify_cifar10_asymmetric(y_train, noise=noise_ratio, random_state=seed)
        elif dataset == 'cifar100':
            noise_y_train, p, keep_indices = noisify_cifar100_asymmetric(y_train, noise=noise_ratio, random_state=seed)
        elif dataset == 'mnist':
            noise_y_train, p, keep_indices = noisify_mnist_asymmetric(y_train, noise=noise_ratio, random_state=seed)
            
    elif noise_type=='flip2':
        noise_y_train, p, keep_indices = flip_labels_C_two(y_train, noise=noise_ratio, nb_classes=num_classes,random_state=seed)
        
    elif noise_type=='inst':
        noise_y_train = noisify_instance(train_data,y_train,num_classes,noise_rate=noise_ratio,random_state=seed )
                       
    elif 'human' in noise_type:
        if dataset == 'cifar10':
            noise_file_name='CIFAR-10_human.pt'
        elif dataset == 'cifar100':
            noise_file_name='CIFAR-100_human.pt'
            
        noise_path = os.path.join(data_dir,dataset,noise_file_name)
        noise_file= torch.load(noise_path)
        if dataset == 'cifar10':
            if 'worst' in noise_type:
                noisy_label = noise_file['worse_label']
            elif 'aggre' in noise_type:
                noisy_label = noise_file['aggre_label']
            elif 'random' in noise_type:
                noisy_label = noise_file['random_label1']
            
        elif dataset == 'cifar100':
            noisy_label = noise_file['noisy_label']
            
        # if lb_ids.shape[0]==50000:
        #     return noisy_label     

        
    # torch.save(noise_y_train,lbs_file_path)
    print('already save noisy label file.')
    return noisy_label
