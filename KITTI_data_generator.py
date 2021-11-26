import glob
import os
import sys
from pathlib import Path
import random

try:
    sys.path.append(glob.glob('%s/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        "C:/CARLA_0.9.10/WindowsNoEditor" if os.name == 'nt' else str(Path.home()) + "/CARLA_0.9.10",
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import time
from datetime import date
from modules import generator_KITTI as gen

def main():
    start_record_full = time.time()

    fps_simu = 1000.0
    time_stop = 2.0
    nbr_frame = 5000 #MAX = 10000
    nbr_walkers = 50
    nbr_vehicles = 50

    actor_list = []
    vehicles_list = []
    all_walkers_id = []
    data_date = date.today().strftime("%Y_%m_%d")
    
    spawn_points = [23,46,0,125,53,257,62]
    
    init_settings = None

    try:
        client = carla.Client('localhost', 2000)
        init_settings = carla.WorldSettings()
        
        for i_map in [0, 1, 2, 3, 4, 5, 6]: #7 maps from Town01 to Town07
            client.set_timeout(100.0)
            print("Map Town0"+str(i_map+1))
            world = client.load_world("Town0"+str(i_map+1))
            folder_output = "KITTI_Dataset_CARLA_v%s/%s/generated" %(client.get_client_version(), world.get_map().name)
            os.makedirs(folder_output) if not os.path.exists(folder_output) else [os.remove(f) for f in glob.glob(folder_output+"/*") if os.path.isfile(f)]
            client.start_recorder(os.path.dirname(os.path.realpath(__file__))+"/"+folder_output+"/recording.log")
            
            # Weather
            world.set_weather(carla.WeatherParameters.WetCloudyNoon)
            
            # Set Synchronous mode
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0/fps_simu
            settings.no_rendering_mode = False
            world.apply_settings(settings)

            # Create KITTI vehicle
            blueprint_library = world.get_blueprint_library()
            bp_KITTI = blueprint_library.find('vehicle.tesla.model3')
            bp_KITTI.set_attribute('color', '228, 239, 241')
            bp_KITTI.set_attribute('role_name', 'KITTI')
            start_pose = world.get_map().get_spawn_points()[spawn_points[i_map]]
            KITTI = world.spawn_actor(bp_KITTI, start_pose)
            waypoint = world.get_map().get_waypoint(start_pose.location)
            actor_list.append(KITTI)
            print('Created %s' % KITTI)

            # Spawn vehicles and walkers
            gen.spawn_npc(client, nbr_vehicles, nbr_walkers, vehicles_list, all_walkers_id)

            # Wait for KITTI to stop
            start = world.get_snapshot().timestamp.elapsed_seconds
            print("Waiting for KITTI to stop ...")
            while world.get_snapshot().timestamp.elapsed_seconds-start < time_stop: world.tick()
            print("KITTI stopped")

            # Set sensors transformation from KITTI
            lidar_transform     = carla.Transform(carla.Location(x=0, y=0, z=1.80), carla.Rotation(pitch=0, yaw=180, roll=0))
            cam0_transform = carla.Transform(carla.Location(x=0.30, y=0, z=1.70), carla.Rotation(pitch=0, yaw=0, roll=0))
            cam1_transform = carla.Transform(carla.Location(x=0.30, y=0.50, z=1.70), carla.Rotation(pitch=0, yaw=0, roll=0))

            # Take a screenshot
            gen.screenshot(KITTI, world, actor_list, folder_output, carla.Transform(carla.Location(x=0.0, y=0, z=2.0), carla.Rotation(pitch=0, yaw=0, roll=0)))

            # Create our sensors
            gen.RGB.sensor_id_glob = 0
            gen.SS.sensor_id_glob = 10
            gen.Depth.sensor_id_glob = 20
            gen.HDL64E.sensor_id_glob = 100
            VelodyneHDL64 = gen.HDL64E(KITTI, world, actor_list, folder_output, lidar_transform)
            cam0 = gen.RGB(KITTI, world, actor_list, folder_output, cam0_transform)
            cam1 = gen.RGB(KITTI, world, actor_list, folder_output, cam1_transform)
            cam0_ss = gen.SS(KITTI, world, actor_list, folder_output, cam0_transform)
            cam1_ss = gen.SS(KITTI, world, actor_list, folder_output, cam1_transform)
            cam0_depth = gen.Depth(KITTI, world, actor_list, folder_output, cam0_transform)
            cam1_depth = gen.Depth(KITTI, world, actor_list, folder_output, cam1_transform)

            # Export LiDAR to cam0 transformation
            tf_lidar_cam0 = gen.transform_lidar_to_camera(lidar_transform, cam0_transform)
            with open(folder_output+"/lidar_to_cam0.txt", 'w') as posfile:
                posfile.write("#R(0,0) R(0,1) R(0,2) t(0) R(1,0) R(1,1) R(1,2) t(1) R(2,0) R(2,1) R(2,2) t(2)\n")
                posfile.write(str(tf_lidar_cam0[0][0])+" "+str(tf_lidar_cam0[0][1])+" "+str(tf_lidar_cam0[0][2])+" "+str(tf_lidar_cam0[0][3])+" ")
                posfile.write(str(tf_lidar_cam0[1][0])+" "+str(tf_lidar_cam0[1][1])+" "+str(tf_lidar_cam0[1][2])+" "+str(tf_lidar_cam0[1][3])+" ")
                posfile.write(str(tf_lidar_cam0[2][0])+" "+str(tf_lidar_cam0[2][1])+" "+str(tf_lidar_cam0[2][2])+" "+str(tf_lidar_cam0[2][3]))
            
            # Export LiDAR to cam1 transformation
            tf_lidar_cam1 = gen.transform_lidar_to_camera(lidar_transform, cam1_transform)
            with open(folder_output+"/lidar_to_cam1.txt", 'w') as posfile:
                posfile.write("#R(0,0) R(0,1) R(0,2) t(0) R(1,0) R(1,1) R(1,2) t(1) R(2,0) R(2,1) R(2,2) t(2)\n")
                posfile.write(str(tf_lidar_cam1[0][0])+" "+str(tf_lidar_cam1[0][1])+" "+str(tf_lidar_cam1[0][2])+" "+str(tf_lidar_cam1[0][3])+" ")
                posfile.write(str(tf_lidar_cam1[1][0])+" "+str(tf_lidar_cam1[1][1])+" "+str(tf_lidar_cam1[1][2])+" "+str(tf_lidar_cam1[1][3])+" ")
                posfile.write(str(tf_lidar_cam1[2][0])+" "+str(tf_lidar_cam1[2][1])+" "+str(tf_lidar_cam1[2][2])+" "+str(tf_lidar_cam1[2][3]))


            # Launch KITTI
            KITTI.set_autopilot(True)

            # Pass to the next simulator frame to spawn sensors and to retrieve first data
            world.tick()
            
            VelodyneHDL64.init()
            gen.follow(KITTI.get_transform(), world)
            
            # All sensors produce first data at the same time (this ts)
            gen.Sensor.initial_ts = world.get_snapshot().timestamp.elapsed_seconds
            
            start_record = time.time()
            print("Start record : ")
            frame_current = 0
            while (frame_current < nbr_frame):
                frame_current = VelodyneHDL64.save()
                cam0.save()
                cam1.save()
                cam0_ss.save()
                cam1_ss.save()
                cam0_depth.save()
                cam1_depth.save()
                gen.follow(KITTI.get_transform(), world)
                world.tick()    # Pass to the next simulator frame
            
            VelodyneHDL64.save_poses()
            client.stop_recorder()
            print("Stop record")
            
            print('Destroying %d vehicles' % len(vehicles_list))
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
            vehicles_list.clear()
            
            # Stop walker controllers (list is [controller, actor, controller, actor ...])
            all_actors = world.get_actors(all_walkers_id)
            for i in range(0, len(all_walkers_id), 2):
                all_actors[i].stop()
            print('Destroying %d walkers' % (len(all_walkers_id)//2))
            client.apply_batch([carla.command.DestroyActor(x) for x in all_walkers_id])
            all_walkers_id.clear()
                
            print('Destroying KITTI')
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
            actor_list.clear()
                
            print("Elapsed time : ", time.time()-start_record)
            print()
                
            time.sleep(2.0)

    finally:
        print("Elapsed total time : ", time.time()-start_record_full)
        world.apply_settings(init_settings)
        
        time.sleep(2.0)
        

if __name__ == '__main__':
    main()
