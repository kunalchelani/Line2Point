import numpy as np
from colmap import read_model
import random
from scipy.spatial import cKDTree
from tqdm import tqdm
import open3d as o3d

def get_n_closest_lines_from_line(pt, line, pts, lines, num_nn):

    n = np.cross(line, lines)
    n /= np.linalg.norm(n, axis = 1, keepdims = True) + 10e-7

    dist = np.abs(np.sum(np.multiply(pts - pt, n), axis = 1))
    ii_nn = np.argpartition(dist, (0,num_nn+1))

    return ii_nn[1:num_nn+1]

def get_n_closest_points_from_line(pt, line, pts, num_nn):

    n = np.cross(pts - pt, line)
    n /= np.linalg.norm(n, axis = 1, keepdims = True) + 10e-7

    n1 = np.cross(n, line)
    n1 /= np.linalg.norm(n1, axis = 1, keepdims = True) + 10e-7

    dist = np.abs(np.sum(np.multiply(pts - pt, n1), axis = 1))
    ii_nn = np.argpartition(dist, (0,num_nn+1))

    return ii_nn[1:num_nn+1]

def get_n_closest_lines_from_point(pt, pts, lines, num_nn):

    n = np.cross(pts - pt, lines)
    n /= np.linalg.norm(n, axis = 1, keepdims = True) + 10e-7

    n1 = np.cross(n, lines)
    n1 /= np.linalg.norm(n1, axis = 1, keepdims = True) + 10e-7

    dist = np.abs(np.sum(np.multiply(pts - pt, n1), axis = 1))
    ii_nn = np.argpartition(dist, (0,num_nn+1))

    return ii_nn[1:num_nn+1]

def calc_line_line_dist(p1, l1, p2, l2):
    n = np.cross(l1, l2)
    nm = np.linalg.norm(n)
    if nm < 0.01:
        return 1000

    n /= nm
    dist = np.abs(np.dot(p2 - p1, n))
    return dist

def calc_estimates_from_lines(pt, line, neigh_pts, neigh_lines):
    ests = []
    for i in range(neigh_lines.shape[0]):

        est = calc_estimate_from_line(pt, line, neigh_pts[i, :], neigh_lines[i, :])
        ests.append(est)

    return ests

def calc_estimate_from_line(pt_est, line_est, pt_use, line_use):

    n = np.cross(line_est, line_use)
    n /= np.linalg.norm(n) + 10e-7

    n2 = np.cross(n, line_use)
    n2 /= np.linalg.norm(n2) + 10e-7

    est = np.dot((pt_est  - pt_use),n2)/(np.dot(line_est, n2) + 10e-7)
    return est

def estimate_all_pts(pts, lines, nns):

    num_pts = pts.shape[0]
    estimates = {}
    estimates_peak = np.zeros([num_pts])

    for i in tqdm(range(num_pts)):
        nn = nns[i, :]
        estimates[i] = calc_estimates_from_lines(pts[i,:], lines[i,:], pts[nn,:], lines[nn, :])

    print("Finding peaks.")
    for i in range(num_pts):
        estimates_peak[i], _ = get_peak(np.array(estimates[i]))

    return estimates_peak

def estimate_all_pts_dict(pts, lines, nns):

    num_pts = pts.shape[0]
    estimates = {}
    estimates_peak = np.zeros([num_pts])

    for i in tqdm(range(num_pts)):
        nn = nns[i]
        estimates[i] = calc_estimates_from_lines(pts[i,:], lines[i,:], pts[nn,:], lines[nn, :])

    print("Finding peaks.")
    for i in range(num_pts):
        estimates_peak[i], _ = get_peak(np.array(estimates[i]))

    return estimates_peak

def get_peak(estimates, num_bins = 500, nro = 5):
    est_clean = np.sort(estimates)[nro:-1*nro]
    kv = 1
    max_kv = 0
    while est_clean.shape[0] > 5 and kv > 0.3:
        in_peak, kv = find_peak(est_clean, num_bins)
        est_clean = est_clean[in_peak]
        if kv > max_kv:
            max_kv = kv
    if(est_clean.shape[0] == 0):
        peak = np.median(estimates)
    else:
        peak = np.median(est_clean)
    return peak, max_kv

def find_peak(estimates, num_bins = 500):
    # Implement the Kolmogorov-Smirnov and Kuiper's here
    hist_cs = np.zeros(num_bins + 1)
    uni_cs = np.zeros(num_bins + 1)

    uni = np.ones(num_bins)/num_bins
    hist, edges = np.histogram(estimates, bins = num_bins)

    hist_cs[1:] = np.cumsum(hist)/np.sum(hist)
    uni_cs[1:] = np.cumsum(uni)/np.sum(uni)

    #plt.plot(edges, hist_cs)
    #plt.plot(edges, uni_cs)
    min_diff_ind = np.argmin(hist_cs[0:int(0.9 *num_bins)] - uni_cs[0:int(0.9 *num_bins)])
    max_diff_ind = np.argmax(hist_cs[min_diff_ind:-1] - uni_cs[min_diff_ind:-1]) + min_diff_ind
    #plt.axvline(x = edges[min_diff_ind], color = 'red')
    #plt.axvline(x = edges[max_diff_ind], color = 'red')
    kuipers_value = hist_cs[max_diff_ind] + uni_cs[min_diff_ind] - hist_cs[min_diff_ind] - uni_cs[max_diff_ind]
    #plt.show()

    in_peak = (estimates < edges[max_diff_ind]) & (estimates > edges[min_diff_ind])
    return in_peak, kuipers_value # Returns the indices of estimates within peak and kuipers's statistic

def load_points_and_setup(points_fname, use_fraction):

    # Read file
    points3D = read_model.read_points3d_text(points_fname)
    num_pts = len(points3D)


    # Prepare indices that need to be selected
    num_pts_sample = int(use_fraction * num_pts)
    sel_ids = random.sample(list(points3D.keys()), k = num_pts_sample)

    j = 0

    ind_to_id = {}
    id_to_ind = {}

    for pt_id in sel_ids:

        ind_to_id[j] = pt_id
        id_to_ind[pt_id] = j
        j += 1

    pts = np.zeros([num_pts_sample, 3])
    lines = np.random.randn(num_pts_sample, 3)
    lines /= np.linalg.norm(lines, axis = 1, keepdims = True)

    for k in range(num_pts_sample):
        pts[k,:] = points3D[ind_to_id[k]].xyz

    return points3D, pts, lines, ind_to_id, id_to_ind

def write_colmap_points(points_fname_out, points, estimates, if_use_pt, id_to_ind):

    num_pts = len(if_use_pt)

    with open(points_fname_out, "w") as fid:

        fid.write("# 3D point list with one line of data per point:\n\
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        fid.write("#\n")

        for i in points.keys():

            #id = ind_to_id[i]

            if(if_use_pt[i]):

                ind = id_to_ind[i] 

                fid.write("{} ".format(i))
                fid.write("{} {} {} ".format(estimates[ind, 0], estimates[ind, 1], estimates[ind, 2]))
                fid.write("{} {} {} ".format(points[i].rgb[0], points[i].rgb[1], points[i].rgb[2]))
                fid.write("{} ".format(points[i].error))

                track_len = len(points[i].image_ids)

                for j in range(track_len):

                    fid.write("{} {} ".format(points[i].image_ids[j], points[i].point2D_idxs[j]))

                fid.write("\n")

def write_colmap_images(images_fname_out, images_fname_in, if_use_pt):

    images = read_model.read_images_text(images_fname_in)


    with open(images_fname_out, "w") as fid:
        fid.write("# Image list with two lines of data per image:\n\
# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME \n\
# POINTS2D[] as (X, Y, POINT3D_ID)\n## \n")

        for image in images.values():
            fid.write("{} {} {} {} {} ".format(image.id, image.qvec[0], image.qvec[1], image.qvec[2], image.qvec[3]))
            fid.write("{} {} {} ".format(image.tvec[0], image.tvec[1], image.tvec[2]))
            fid.write("{} {}\n".format(image.camera_id, image.name))
            for i in range(len(image.point3D_ids)):
                if (if_use_pt[image.point3D_ids[i]]):
                    fid.write("{} {} {} ".format(image.xys[i, 0], image.xys[i, 1],image.point3D_ids[i]))
            fid.write("\n")

def filter_and_write_ply(pts_in_fname, ply_fname, use_fraction, nn1, std1, nn2, std2):
    print("Loading points and setting up.")
    Points, pts, lines, ind_to_id, id_to_ind = load_points_and_setup(pts_in_fname, use_fraction)
    print("Done.")

    num_pts = pts.shape[0]
    print(num_pts)
    if_use_pt = {}
    if_use_pt[-1] = False
    for i in range(num_pts):
        if_use_pt[ ind_to_id[i] ] = False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors = int(nn1), std_ratio = float(std1))
    pcd2 = pcd.select_by_index(ind)

    cl2, ind2 = pcd2.remove_statistical_outlier(nb_neighbors = int(nn2), std_ratio = float(std2))
    pcd3 = pcd2.select_by_index(ind2)

    for i in range(len(ind2)):
        inlier = ind[ind2[i]]
        inlier_id = ind_to_id[ inlier ]
        if_use_pt[ inlier_id ] = True

    print(len(ind2))
    o3d.io.write_point_cloud(ply_fname, pcd3)

def filter_thrice_and_write_ply(pts_in_fname, ply_fname, use_fraction, nn1, std1, nn2, std2, nn3, std3):
    print("Loading points and setting up.")
    Points, pts, lines, ind_to_id, id_to_ind = load_points_and_setup(pts_in_fname, use_fraction)
    print("Done.")

    num_pts = pts.shape[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors = int(nn1), std_ratio = float(std1))
    pcd2 = pcd.select_by_index(ind)

    cl2, ind2 = pcd2.remove_statistical_outlier(nb_neighbors = int(nn2), std_ratio = float(std2))
    pcd3 = pcd2.select_by_index(ind2)

    cl3, ind3 = pcd3.remove_statistical_outlier(nb_neighbors = int(nn3), std_ratio = float(std3))
    pcd4 = pcd3.select_by_index(ind3)

    print(len(ind))
    print(len(ind2))
    print(len(ind3))

    o3d.io.write_point_cloud(ply_fname, pcd4)
'''
def write_line_directions(lines_fname_out, lines, ind_to_id):
    num_pts = lines.shape[0]
    with open(lines_fname_out, 'w'):
        for i in range(num_pts):
            ind_to_id[i]
            lines(
'''

def filter_and_write_colmap(pts_in_fname, im_in_fname, use_fraction, nn1, std1, nn2, std2):

    print("Loading points and setting up.")
    Points, pts, lines, ind_to_id, id_to_ind = load_points_and_setup(pts_in_fname, use_fraction)
    print("Done.")

    num_pts = pts.shape[0]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)

    if_use_pt = {}
    if_use_pt[-1] = False
    for i in range(num_pts):
        if_use_pt[ ind_to_id[i] ] = False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors = int(nn1), std_ratio = float(std1))
    pcd2 = pcd.select_by_index(ind)

    cl2, ind2 = pcd2.remove_statistical_outlier(nb_neighbors = int(nn2), std_ratio = float(std2))
    pcd3 = pcd2.select_by_index(ind2)

    for i in range(len(ind2)):
        inlier = ind[ind2[i]]
        inlier_id = ind_to_id[ inlier ]
        if_use_pt[ inlier_id ] = True

    points_fname_out = pts_in_fname.split('.')[0] + "_filter" + ".txt"
    images_fname_out = im_in_fname.split('.')[0] + "_filter" + ".txt"

    write_colmap_points(points_fname_out, Points, pts, if_use_pt, id_to_ind)
    write_colmap_images(images_fname_out, im_in_fname, if_use_pt)


def write_linecloud(pts_in_fname, lc_out_fname, line_seg_length, use_fraction):
    print("Loading points and setting up.")
    Points, pts, lines, ind_to_id, id_to_ind = load_points_and_setup(pts_in_fname, use_fraction)
    print("Done.")
    end_pts_1 = pts - line_seg_length * lines
    end_pts_2 = pts + line_seg_length * lines

    with open(lc_out_fname, 'w') as fid:
        fid.write("# 3D line cloud\n")
        fid.write("# vertices\n")
        for i in range(pts.shape[0]):
            fid.write("v {} {} {}\n".format(end_pts_1[i,0], end_pts_1[i,1], end_pts_1[i,2]))
            fid.write("v {} {} {}\n".format(end_pts_2[i,0], end_pts_2[i,1], end_pts_2[i,2]))
        for i in range(pts.shape[0]):
            fid.write("l {} {}\n".format(int(2*i+1), int(2*i+2)))

def write_ests_to_ply(pts_est, ply_fname):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_est)
    o3d.io.write_point_cloud(ply_fname, pcd)