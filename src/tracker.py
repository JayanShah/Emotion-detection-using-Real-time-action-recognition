import numpy as np
import math

class SimpleTracker:
    def __init__(self, max_dist=50):
        self.center_points = {}  # id: (cx, cy)
        self.id_count = 0
        self.max_dist = max_dist

    def update(self, pose_keypoints):
        objects_bbs = []
        for i in range(pose_keypoints.shape[0]):
            valid_points = pose_keypoints[i, :, :2]
            valid_points = valid_points[valid_points[:, 0] > 0]
            if len(valid_points) > 0:
                cx = int(np.mean(valid_points[:, 0]))
                cy = int(np.mean(valid_points[:, 1]))
                objects_bbs.append([cx, cy, i])

        new_center_points = {}
        result_mapping = {}

        for obj_bb in objects_bbs:
            cx, cy, idx = obj_bb
            same_object_detected = False
            for object_id, center in self.center_points.items():
                dist = math.hypot(cx - center[0], cy - center[1])
                if dist < self.max_dist:
                    self.center_points.pop(object_id)
                    new_center_points[object_id] = (cx, cy)
                    result_mapping[idx] = object_id
                    same_object_detected = True
                    break

            if not same_object_detected:
                new_center_points[self.id_count] = (cx, cy)
                result_mapping[idx] = self.id_count
                self.id_count += 1

        self.center_points = new_center_points.copy()
        return result_mapping