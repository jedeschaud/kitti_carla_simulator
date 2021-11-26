import glob
import os
import numpy as np
import sys
import time

from modules import ply


SIZE_POINT_CLOUD = 10000000

DETECT_STOP_VEHICLE = True
STOP_VEHICLE_VELOCITY_NORM = 0.3 #same parameter as in npm_georef for L3D2

ADD_LIDAR_NOISE = False
SIGMA_LIDAR_NOISE = 0.0 #0.03 = std dev of the Velodyne HDL64E

ROOT_DIR = "KITTI_Dataset_CARLA_v993f440b/Town01"


def main():

    folder_input = ROOT_DIR+"/generated"
    folder_output = ROOT_DIR+"/georef"

    # Create folders or remove files
    os.makedirs(folder_output) if not os.path.exists(folder_output) else [os.remove(f) for f in glob.glob(folder_output+"/*") if os.path.isfile(f)]

    file_id_point_cloud = 0
    point_cloud = np.empty([SIZE_POINT_CLOUD,8], dtype = np.float32)
    point_cloud_index_semantic = np.empty([SIZE_POINT_CLOUD,3], dtype = np.uint32)
    index_point_cloud = 0

    # Open file and read header
    poses_file = open(folder_input+"/full_poses_lidar.txt", 'r')
    poses_file.readline()
    line_pose = np.array(poses_file.readline().split(), float)
    tf = np.vstack((line_pose[:-1].reshape(3,4), [0,0,0,1]))
    ts_tf = line_pose[-1]
    previous_tf = tf
    previous_ts_tf = ts_tf

    ply_files = sorted(glob.glob(folder_input+"/frames/frame*.ply"))
    
    start_record = time.time()
    max_velocity = 0.
    index_ply_file = 0
    for f in ply_files:
        data = ply.read_ply(f)
        nbr_pts = len(data)

        i = 0
        while (i < nbr_pts):
        
            while (np.abs(data[i]['timestamp']-ts_tf)>1e-4):
                line_pose = np.array(poses_file.readline().split(), float)
                previous_tf = tf
                tf = np.vstack((line_pose[:-1].reshape(3,4), [0,0,0,1]))
                previous_ts_tf = ts_tf
                ts_tf = line_pose[-1]
            
            if (np.abs(data[i]['timestamp']-ts_tf)>1e-4):
                print("Error in timestamp")
                sys.exit()
                
            last_point = i
            while ((last_point<nbr_pts) and (np.abs(data[last_point]['timestamp']-ts_tf)<=1e-4)):
                last_point +=1
            
            current_velocity = 0.4
            if (previous_ts_tf !=ts_tf):
                current_velocity = np.linalg.norm(tf[:-1,3] - previous_tf[:-1,3])/(ts_tf - previous_ts_tf)
            if (current_velocity > max_velocity):
                max_velocity = current_velocity
            
            if (not(DETECT_STOP_VEHICLE) or (current_velocity > STOP_VEHICLE_VELOCITY_NORM)):
                
                pts = np.vstack(np.array([data[i:last_point]['x'], data[i:last_point]['y'], data[i:last_point]['z'], np.full(last_point-i, 1)]))
                new_pts = tf.dot(pts).T
                
                if (ADD_LIDAR_NOISE):
                    vector_pose_to_new_pts = new_pts[:,:-1] - tf[:-1,3]
                    new_pts[:,0] = new_pts[:,0] + np.random.randn(last_point-i)*SIGMA_LIDAR_NOISE*vector_pose_to_new_pts[:,0]/np.linalg.norm(vector_pose_to_new_pts, axis=1)
                    new_pts[:,1] = new_pts[:,1] + np.random.randn(last_point-i)*SIGMA_LIDAR_NOISE*vector_pose_to_new_pts[:,1]/np.linalg.norm(vector_pose_to_new_pts, axis=1)
                    new_pts[:,2] = new_pts[:,2] + np.random.randn(last_point-i)*SIGMA_LIDAR_NOISE*vector_pose_to_new_pts[:,2]/np.linalg.norm(vector_pose_to_new_pts, axis=1)

                if ((index_point_cloud+last_point-i) < SIZE_POINT_CLOUD):
                    point_cloud[index_point_cloud:index_point_cloud+last_point-i,:] = np.vstack([new_pts[:, 0], new_pts[:, 1], new_pts[:, 2], np.full(last_point-i, tf[0,3]), np.full(last_point-i, tf[1,3]), np.full(last_point-i, tf[2,3]), data[i:last_point]['cos_angle_lidar_surface'], data[i:last_point]['timestamp']]).T
                    point_cloud_index_semantic[index_point_cloud:index_point_cloud+last_point-i,:] = np.vstack([np.full(last_point-i, index_ply_file), data[i:last_point]['instance'], data[i:last_point]['semantic']]).T
                    index_point_cloud += last_point-i
                else:
                    last_temp_point = SIZE_POINT_CLOUD - index_point_cloud + i
                    point_cloud[index_point_cloud:index_point_cloud+last_temp_point-i,:] = np.vstack([new_pts[0:last_temp_point-i, 0], new_pts[0:last_temp_point-i, 1], new_pts[0:last_temp_point-i, 2], np.full(last_temp_point-i, tf[0,3]), np.full(last_temp_point-i, tf[1,3]), np.full(last_temp_point-i, tf[2,3]), data[i:last_temp_point]['cos_angle_lidar_surface'], data[i:last_temp_point]['timestamp']]).T
                    point_cloud_index_semantic[index_point_cloud:index_point_cloud+last_temp_point-i,:] = np.vstack([np.full(last_temp_point-i, index_ply_file), data[i:last_temp_point]['instance'], data[i:last_temp_point]['semantic']]).T
                    
                    field_names = ['x','y','z','x_sensor_position','y_sensor_position','z_sensor_position','cos_angle_lidar_surface','timestamp','index_frame','instance','semantic']
                    ply_file_path = folder_output+"/point_cloud_%02d.ply"%file_id_point_cloud
                    file_id_point_cloud += 1
                    if ply.write_ply(ply_file_path, [point_cloud, point_cloud_index_semantic], field_names):
                        print("Export : "+ply_file_path)
                    else:
                        print('ply.write_ply() failed')
                    
                    index_point_cloud = 0
                    point_cloud[index_point_cloud:index_point_cloud+last_point-last_temp_point,:] = np.vstack([new_pts[last_temp_point-i:last_point-i, 0], new_pts[last_temp_point-i:last_point-i, 1], new_pts[last_temp_point-i:last_point-i, 2], np.full(last_point-last_temp_point, tf[0,3]), np.full(last_point-last_temp_point, tf[1,3]), np.full(last_point-last_temp_point, tf[2,3]), data[last_temp_point:last_point]['cos_angle_lidar_surface'], data[last_temp_point:last_point]['timestamp']]).T
                    point_cloud_index_semantic[index_point_cloud:index_point_cloud+last_point-last_temp_point,:] = np.vstack([np.full(last_point-last_temp_point, index_ply_file), data[last_temp_point:last_point]['instance'], data[last_temp_point:last_point]['semantic']]).T
                    index_point_cloud = last_point-last_temp_point
                    
            i = last_point
            
        index_ply_file +=1
    
    print("time.time()-start_record : ", time.time()-start_record)
    print("Max velocity : ", max_velocity, " m/s")
    poses_file.close()

if __name__ == '__main__':
    main()
