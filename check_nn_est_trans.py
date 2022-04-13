import numpy as np
from tqdm import tqdm

def get_n_closest_lines_from_line(pt, line, pts, lines, num_nn):
    
    n = np.cross(line, lines)
    n /= np.linalg.norm(n, axis = 1, keepdims = True) + 10e-7

    dist = np.abs(np.sum(np.multiply(pts - pt, n), axis = 1))
    ii_nn = np.argpartition(dist, num_nn)

    return ii_nn[0:num_nn]


def calc_estimate_from_line(pt_est, line_est, pt_use, line_use):
    
    n = np.cross(line_est, line_use)
    n /= np.linalg.norm(n) + 10e-7

    n2 = np.cross(n, line_use)
    n2 /= np.linalg.norm(n2) + 10e-7

    est = np.dot((pt_est  - pt_use),n2)/(np.dot(line_est, n2) + 10e-7)
    return est

def calc_estimate_from_line_rectified(pt_est, line_est, pt_use, line_use):
    
    n = np.cross(line_est, line_use)
    n /= np.linalg.norm(n) + 10e-7

    n2 = np.cross(n, line_use)
    n2 /= np.linalg.norm(n2) + 10e-7

    est = np.dot((pt_use -pt_est),n2)/(np.dot(line_est, n2) + 10e-7)
    return est

def calc_estimates_from_lines(pt, line, neigh_pts, neigh_lines):
    ests = []
    for i in range(neigh_lines.shape[0]):
        if rev is False:
            est = calc_estimate_from_line(pt, line, neigh_pts[i, :], neigh_lines[i, :])
        if rev is True:
            est = calc_estimate_from_line_rectified(pt, line, neigh_pts[i, :], neigh_lines[i, :])
        ests.append(est)

    return ests


def estimate_all_pts(pts, lines, nns):
    num_pts = pts.shape[0]
    estimates = {}
    estimates_peak = np.zeros([num_pts])

    for i in tqdm(range(num_pts)):
        nn = nns[i, :]
        estimates[i] = calc_estimates_from_lines(pts[i, :], lines[i, :], pts[nn, :], lines[nn, :])

    print("Finding peaks.")

    for i in range(num_pts):
        estimates_peak[i], _ = get_peak(np.array(estimates[i]))

    return estimates_peak


def find_peak(estimates, num_bins=500):
    # Implement the Kolmogorov-Smirnov and Kuiper's here
    hist_cs = np.zeros(num_bins + 1)
    uni_cs = np.zeros(num_bins + 1)

    uni = np.ones(num_bins) / num_bins
    hist, edges = np.histogram(estimates, bins=num_bins)

    hist_cs[1:] = np.cumsum(hist) / np.sum(hist)
    uni_cs[1:] = np.cumsum(uni) / np.sum(uni)

    min_diff_ind = np.argmin(hist_cs[0:int(0.9 * num_bins)] - uni_cs[0:int(0.9 * num_bins)])
    max_diff_ind = np.argmax(hist_cs[min_diff_ind:-1] - uni_cs[min_diff_ind:-1]) + min_diff_ind
    kuipers_value = hist_cs[max_diff_ind] + uni_cs[min_diff_ind] - hist_cs[min_diff_ind] - uni_cs[max_diff_ind]

    in_peak = (estimates < edges[max_diff_ind]) & (estimates > edges[min_diff_ind])
    return in_peak, kuipers_value  # Returns the indices of estimates within peak and kuipers's statistic


def get_peak(estimates, num_bins=500, nro=5):
    est_clean = np.sort(estimates)[nro:-1 * nro]
    kv = 1
    max_kv = 0
    while est_clean.shape[0] > 5 and kv > 0.3:
        in_peak, kv = find_peak(est_clean, num_bins)
        est_clean = est_clean[in_peak]
        if kv > max_kv:
            max_kv = kv

    if (est_clean.shape[0] == 0):
        peak = np.median(estimates)
    else:
        peak = np.median(est_clean)
    return peak, max_kv


##############################################################################
##########################          Main         #############################
##############################################################################

bin_pcd = np.fromfile("demo.bin", dtype=np.float32)
points3d = bin_pcd.reshape((-1, 3))[:2000] # Used only 2000 for fast calculation

print(len(points3d), "Point Found ! ")
pts = points3d[:]
num_pts = pts.shape[0]
num_bins = 500

lines = np.random.randn(num_pts, 3)  
lines /= np.linalg.norm(lines, axis=1, keepdims=True)

num_nn_l2l = int(min(500, 0.05 * num_pts))

nn_l2l = np.zeros([num_pts, num_nn_l2l], dtype=np.int32)
print("Calculating line neighbours")
for i in range(num_pts):
    nn_l2l[i] = get_n_closest_lines_from_line(pts[i], lines[i], pts, lines, num_nn_l2l)

for i in range(2):
    if i==0:
        rev = False
        print('Before revised')
    else:
        rev =  True
        print('After revised')
    sort_nn_org=np.sort(get_n_closest_lines_from_line(pts[0], lines[0], pts, lines, num_nn_l2l))
    sort_nn_trans=np.sort(get_n_closest_lines_from_line(pts[0]+np.multiply(lines[0],20), lines[0], pts, lines, num_nn_l2l))
    print('NN correspondense check :',np.equal(sort_nn_trans,sort_nn_org).all())

    pt0_org_est,_ = get_peak(np.array(calc_estimates_from_lines(pts[0], lines[0], pts[nn_l2l[0]], lines[nn_l2l[0]])))
    pt0_trans_est,_ = get_peak(np.array(calc_estimates_from_lines(pts[0]+np.multiply(lines[0],10), lines[0], pts[nn_l2l[0]], lines[nn_l2l[0]])))

    print('pts :',pts[0])
    print('pts_est :',pts[0]+np.multiply(pt0_org_est,lines[0]))
    print('pst_translated :',pts[0]+np.multiply(lines[0],10))
    print('pts_trans_est :',pts[0]+np.multiply(lines[0],10)+np.multiply(pt0_trans_est,lines[0]))
    print('#'*30)
