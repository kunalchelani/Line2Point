import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
from colmap import *
from Utils import *


parser = argparse.ArgumentParser(description='Code for recovering points from Lines')

# Input
parser.add_argument("--input_type", type = str, default = "colmap", choices=["colmap", "ply"], help="If colmap is chosesn, arguments colmap_input_img_fname, colmap_input_pts_fname need to be provided as well", required=True)
parser.add_argument("--colmap_input_img_fname", type = str, default = "demo/colmap/input/images.txt", required = False)
parser.add_argument("--colmap_input_pts_fname", type = str, default = "demo/colmap/input/points3D.txt", required = False)
parser.add_argument("--input_ply_fname", type = str, default = "demo/ply/input.ply", required = False)

# Paramaters
parser.add_argument("--use_fraction_pts", type=float, help="Float between 0 and 1 to choose a fraction of points from the input", required = True)
parser.add_argument("--num_refine_iterations", type=int, help="Number of refine iterations - Use 2 or 3")

# Output
parser.add_argument("--write_colmap", type = bool, default = False, required=False, help="Set True only if input type is colmap")
parser.add_argument("--colmap_output_img_fname", type = str, default = "demo/colmap/output/images.txt", required = False)
parser.add_argument("--colmap_output_pts_fname", type = str, default = "demo/colmap/output/points3D.txt", required = False)

parser.add_argument("--write_ply", type = bool, default = False, required=False, help="Can be set True irrespective of input type")
parser.add_argument("--output_ply_fname", type = str, default = "demo/ply/output.ply", required = False)

parser.add_argument("--clean_output", type = bool, help = "Clean noise - nearest neighbor based filtering")

# Setup
args = parser.parse_args()

if (args.input_type == "colmap"):
    print("Loading points and setting up.")
    Points, pts, lines, ind_to_id, id_to_ind = load_points_and_setup(args.colmap_input_pts_fname, args.use_fraction_pts)
    print("Done.")
    print("Starting estimation for {} points".format(pts.shape))

else:

    pts, lines = load_points_and_setup_from_ply(args.input_ply_fname, args.use_fraction_pts)



########### Coarse Estimation #############
print("Finding Coarse Estimates")
num_pts = pts.shape[0]
num_nn_l2l = int(min(500, 0.05 * num_pts))

nn_l2l = np.zeros([num_pts, num_nn_l2l], dtype = np.int32)
print("Calculating line neighbours")
for i in tqdm(range(num_pts)):
    nn_l2l[i,:] = get_n_closest_lines_from_line(pts[i,:], lines[i, :], pts, lines, num_nn_l2l)

est_peak = estimate_all_pts(pts, lines, nn_l2l)

errs = np.abs(est_peak)
print("Coarse estimation done.")
print("Mean error : {}".format(np.mean(errs)) )
print("Median error : {}".format(np.median(errs)) )


pts_est = pts + est_peak.reshape(-1,1) * lines
print(pts_est.shape)

########### Refine Estimation #############

num_refine_iteration = args.num_refine_iterations
num_nn_l2p = 100  # Can be changed and experimented with
num_nn_p2l = 100  # Can be changed and experimented with

for i in range(num_refine_iteration):

    pts_est = pts + est_peak.reshape(-1,1) * lines

    nn_l2p = np.zeros([num_pts, num_nn_l2p], dtype = np.int32)
    nn_p2l = np.zeros([num_pts, num_nn_p2l], dtype = np.int32)

    print("Calculating l2p neighbours")
    for i in range(num_pts):
        nn_l2p[i,:] = get_n_closest_points_from_line(pts[i,:], lines[i, :], pts_est, num_nn_l2p)

    print("Calculating p2l neighbours")
    for i in range(num_pts):
        nn_p2l[i,:] = get_n_closest_lines_from_point(pts_est[i,:], pts, lines, num_nn_p2l)

    refine_estimates = {}
    nns = {}

    print("Finding refined estimates using intersection when possible")
    print(num_pts)
    for i in range(num_pts):

        set_p2l = set(nn_p2l[i,:])
        set_l2p = set(nn_l2p[i,:])
        set_intersection = set_p2l.intersection(set_l2p)

        if len(set_intersection) > 10: # Threshld can be changed as well. A distance metric combining both l2p and p2l can also be defined.
            nns[i] = np.array(list(set_intersection), dtype = np.int32)
        else:
            nns[i] = nn_l2p[i, :]


    est_peak = estimate_all_pts_dict(pts, lines, nns)
    errs = np.abs(est_peak)

    print("Mean error : {}".format(np.mean(errs)))
    print("Median error : {}".format(np.median(errs)))

pts_est = pts + est_peak.reshape(-1,1) * lines

########### Write filtered output #############

# Should make these user input.
nn1 = 50
std1 = 2.0

nn2 = 25
std2 = 2.0

if(args.clean_output):
    num_pts = pts_est.shape[0]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts_est)
    
    if(args.input_type =="colmap" and args.write_colmap):
        if_use_pt = {}
        if_use_pt[-1] = False
        for i in range(num_pts):
            if_use_pt[ ind_to_id[i] ] = False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors = int(nn1), std_ratio = float(std1))
    pcd2 = pcd.select_by_index(ind)

    cl2, ind2 = pcd2.remove_statistical_outlier(nb_neighbors = int(nn2), std_ratio = float(std1))
    pcd3 = pcd2.select_by_index(ind2)

############ Ply ############

    if(args.write_ply):
        o3d.io.write_point_cloud(args.output_ply_fname, pcd3)

############ Colmap ############

    if( (args.input_type == "colmap") & args.write_colmap):

        for i in range(len(ind2)):
            inlier = ind[ind2[i]]
            inlier_id = ind_to_id[ inlier ]
            if_use_pt[ inlier_id ] = True

        write_colmap_points(args.colmap_output_pts_fname, Points, pts_est, if_use_pt, id_to_ind)
        write_colmap_images(args.colmap_input_img_fname, args.colmap_output_img_fname, if_use_pt)



########### Write un-filtered output #############

else:

    if(args.write_ply):
        write_ests_to_ply(pts_est, args.output_ply_fname)

    if( (args.input_type == "colmap") & args.write_colmap):
        if_use_pt = {}
        if_use_pt[-1] = False
        for i in Points.keys():
            if_use_pt[i] = False

        for i in range (num_pts):
            if_use_pt[ind_to_id[i]] = True

        write_colmap_points(args.colmap_output_pts_fname, Points, pts_est, if_use_pt, id_to_ind)
        write_colmap_images(args.colmap_output_img_fname, args.colmap_input_img_fname, if_use_pt)