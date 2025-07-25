import open3d as o3d
import numpy as np
import os
import cv2

class TrackerUtils:
    def __init__(self, intrinsics, window_len=8):
        self.intrinsics = intrinsics
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.cx = intrinsics[0, 2]
        self.cy = intrinsics[1, 2]
        self.window_len = window_len

    def estimate_icp_transform(self, source_points, target_points, threshold=0.05):
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points)

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        return reg_p2p.transformation, reg_p2p.fitness


    def compute_3D_from_mask(self, rgb, mask, depth, cam_pose):
        """Estrai punti 3D da una maschera e immagine di profondità"""
        ys, xs = np.where(mask)
        zs = depth[ys, xs] / 1000.0  # in metri

        valid = zs > 0
        xs, ys, zs = xs[valid], ys[valid], zs[valid]

        X = (xs - self.cx) * zs / self.fx
        Y = (ys - self.cy) * zs / self.fy
        Z = zs

        cam_coords = np.stack([X, Y, Z], axis=1)
        cam_coords_hom = np.concatenate([cam_coords, np.ones((cam_coords.shape[0], 1))], axis=1)
        world_coords = (cam_pose @ cam_coords_hom.T).T[:, :3]
        rgb = rgb.astype(np.float32) / 255.0  # Normalize
        colors = rgb[ys, xs]
        return world_coords, colors

    def estimate_ransac_se3(self, src_points, tgt_points, threshold=0.01):
        src_pcd = o3d.geometry.PointCloud()
        tgt_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_points)
        tgt_pcd.points = o3d.utility.Vector3dVector(tgt_points)

        corres = np.array([[i, i] for i in range(len(src_points))])
        corres = o3d.utility.Vector2iVector(corres)

        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            src_pcd,
            tgt_pcd,
            corres,
            max_correspondence_distance=threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(40000, 500)
        )

        return result.transformation, result.fitness
    

    def align_3D_masks(self, rgb, masks, depths, poses, window_counter=None, refined_3d_points=None, output_dir="merged_pcl"):
        merged_pcd = o3d.geometry.PointCloud()
        os.makedirs(output_dir, exist_ok=True)

        accumulated_T = np.eye(4)

        for t in range(len(rgb) - 1):
            if masks[t] is not None and masks[t+1] is not None:
                src_masks, src_colors = self.compute_3D_from_mask(rgb[t], masks[t], depths[t], poses[t])
                tgt_masks, tgt_colors = self.compute_3D_from_mask(rgb[t+1], masks[t+1], depths[t+1], poses[t+1])
                # src_dict = {id: pt for id, pt in refined_3d_points[t]}
                # tgt_dict = {id: pt for id, pt in refined_3d_points[t + 1]} 

                # common_ids = set(src_dict.keys()) & set(tgt_dict.keys())

                # src_points = np.array([src_dict[i] for i in common_ids])
                # tgt_points = np.array([tgt_dict[i] for i in common_ids])

                # if len(src_points) < 10 or len(tgt_points) < 10:
                #     continue

                T_rel, fitness = self.estimate_icp_transform(src_masks, tgt_masks)
                T_rel_inv = np.linalg.inv(T_rel)
                print(f"Frame {t}->{t+1} | Fitness: {fitness:.3f}")
                print(T_rel)
                # print("Ransasc Transformation:")
                # T_rel_ransac, fitness = self.estimate_ransac_se3(src_points, tgt_points)
                # T_rel_ransac_inv = np.linalg.inv(T_rel_ransac)
                # print(f"Frame {t}->{t+1} | Fitness: {fitness:.3f}")
                # print(T_rel_ransac)

                accumulated_T = accumulated_T @ T_rel_inv 
                # accumulated_T = accumulated_T @ T_rel_ransac_inv 
                tgt_pcd = o3d.geometry.PointCloud()

                
                tgt_pcd.points = o3d.utility.Vector3dVector(tgt_masks)
                tgt_pcd.colors = o3d.utility.Vector3dVector(tgt_colors)
                tgt_pcd.transform(accumulated_T)

                if t==0:
                    src_pcd = o3d.geometry.PointCloud()
                    src_pcd.points = o3d.utility.Vector3dVector(src_masks)
                    src_pcd.colors = o3d.utility.Vector3dVector(src_colors)
                    merged_pcd += src_pcd

                merged_pcd += tgt_pcd

        # Salva o visualizza la nuvola completa
        if window_counter is not None:
            output_path = os.path.join(output_dir, f"merged_frame_{window_counter:04d}.ply")
        else:   
            output_path = os.path.join(output_dir, f"merged_frame_complete.ply")
        o3d.io.write_point_cloud(output_path, merged_pcd)
        # o3d.visualization.draw_geometries([merged_pcd])