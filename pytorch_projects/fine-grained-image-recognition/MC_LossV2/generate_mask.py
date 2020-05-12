import torch
import numpy as np

def _gen_mask(cnums, cgroups, p):
    """
    :cnums = [cnum1, cnum2 ..., cnumc],cnumi表示用于表示第i类的channel数量
    :cgroups = [cgroup1, ...], cgroup表示对应的类别数，比如cnums = [10, 11], cgroups = [152, 48]表示前152类用10个channel表示，后48类用11个channel表示 
    :param p: float, probability of random deactivation
    """
    '''
    cnums = [2, 3], cgroups = [2, 3], p = 0.6
    # cnums[i] * cgroups[i]
    foo = [1, 1, 1, 1]
    # drop_num
    drop_num = 1
    # drop_index
    drop_idx = []
    # np.random.choice(np.arange(2), size=1, replace=False)
    drop_idx = [array([0]), array([2])]
    # 
    '''
    bar = []
    for i in range(len(cnums)):
        foo = np.ones((cgroups[i], cnums[i]), dtype=np.float32).reshape(-1,)
        drop_num = int(cnums[i] * p)
        drop_idx = []
        for j in range(cgroups[i]):
            drop_idx.append(np.random.choice(np.arange(cnums[i]), size=drop_num, replace=False) + j * cnums[i])
        drop_idx = np.stack(drop_idx, axis=0).reshape(-1,)
        foo[drop_idx] = 0.
        bar.append(foo)
    bar = np.hstack(bar).reshape(1, -1, 1, 1)
    bar = torch.from_numpy(bar)

    return bar

if __name__ == "__main__":
    a = _gen_mask([10, 11], [152, 48], 0.4)
    print(a.size())