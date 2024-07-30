import random as rd
import time

# for hamza
def CalDistance(EId):
    mindist = rd.randint(1, 10)
    Cid = rd.randint(1, 3)
    # Cid = 1
    
    return mindist, Cid

#  O(n)
def AssignPairs(employees, customers, Min_Dist_threshold, pairs_dict):
    for employee in employees:
        EId = employee['id']
        minDist, Cid = CalDistance(EId)
        if minDist >= Min_Dist_threshold:
            if (EId, Cid) in pairs_dict:
                # print("added time",pairs_dict[(EId, Cid)])
                pairs_dict[(EId, Cid)][2] += 0.2
                # in here
                pairs_dict[(EId, Cid)][2]=round(pairs_dict[(EId, Cid)][2],2)
                
            else:
                # print("adding")
                pairs_dict[(EId, Cid)] = [EId, Cid, 0.2, 0]
    return pairs_dict

# O(p)
def UpdateFrames(pairs_dict, min_time_threshold, n_frames):
    keys_to_remove = []
    for key, pair in pairs_dict.items():
        EId, CId, time, framesNotUpdated = pair
        if framesNotUpdated < n_frames * min_time_threshold:
            # print("Updating one frame")
            pair[3] += 1
        # if pair[3] == n_frames * min_time_threshold and time < min_time_threshold:
        #     # print("removed in update frame",pair)
        #     keys_to_remove.append(key)
    
    for key in keys_to_remove:
        
        del pairs_dict[key]
    
    return pairs_dict
# O(p)
def DeleteFrames(pairs_dict, min_time_threshold, n_frames):
    keys_to_remove = []
    for key, pair in pairs_dict.items():
        EId, CId, time, framesNotUpdated = pair
        if pair[3] < n_frames * min_time_threshold and time < min_time_threshold:
            # print(pair)
            # print("removed in delete frame",key)
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del pairs_dict[key]
    
    return pairs_dict

# Example usage
employees = [{'id': 1,},
             {'id': 2}, {'id': 3},
             {'id': 4}, {'id': 5}
             
             ]
customers = [{'id': 1}, 
             {'id': 2}, {'id': 3},
             {'id': 4}, {'id': 5}
             ]
Min_Dist_threshold = 1
pairs_dict = {}
start_time = time.time()
for i in range(1, 1000):
    pairs_dict = AssignPairs(employees, customers, Min_Dist_threshold, pairs_dict)
    pairs_dict = UpdateFrames(pairs_dict, min_time_threshold=30, n_frames=5)
pairs_dict = DeleteFrames(pairs_dict, min_time_threshold=30, n_frames=5)
end_time = time.time()
elapsed_time = round(end_time - start_time, 3) * 1000
print(pairs_dict)
print(f"process time: {round(elapsed_time/1000, 3)} s")
