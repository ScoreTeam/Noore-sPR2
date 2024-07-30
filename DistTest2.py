import random as rd

# Define the distance calculation function
def CalDistance(EId):
    # Calculate the minimum distance and corresponding customer ID
    # Placeholder for actual distance calculation logic
    mindist=rd.randint(1,10)
    Cid = rd.randint(1,3)
    # will return the cid with the lowest distance 
    
    return mindist, Cid
def PairExist(pairs,CID,EID):
    for pair in pairs:
        return pair[0] ==EID and pair[1] == CID 
    
# Function to assign pairs
def AssignPairs(employees,customers,Min_Dist_threshold, pairs):
    for employee in employees:
        EId = employee['id']   
        minDist, Cid = CalDistance(EId)
        # print(minDist,Cid)
        if minDist >= Min_Dist_threshold:
            # pair_exists = False
            if PairExist(pairs,EId,Cid):
                for pair in pairs:
                    if pair[0] == EId and pair[1] == Cid:
                        pair[2] +=0.2
                        print("added time",pair)
                        
                        break
            else:
                # Append a new pair
                pairs.append([EId, Cid, 0.2, 0])
    return pairs

# Function to update frames
def UpdateFrames(pairs, min_time_threshold, n_frames):
    new_pairs = []
    for pair in pairs:
        EId, CId, time, framesNotUpdated = list(pair)
        if framesNotUpdated < n_frames*min_time_threshold: 
            framesNotUpdated += 1
        if framesNotUpdated ==  n_frames*min_time_threshold and time < min_time_threshold:
            print("Removed")
            continue  
        new_pairs.append([EId, CId, time, framesNotUpdated])
    return new_pairs

    employees = [
    {'id': 1, 'name': 'John'},
    {'id': 2, 'name': 'Alice'},
    # {'id': 3, 'name': 'Bob'},
    # {'id': 4, 'name': 'Emily'},
    # {'id': 5, 'name': 'Michael'}
    ]
    customers = [
    {'id': 1, 'name': 'Mary'},
    {'id': 2, 'name': 'David'},
    {'id': 3, 'name': 'Sarah'},
    {'id': 4, 'name': 'Emma'},
    {'id': 5, 'name': 'James'}
]
    Min_Dist_threshold=5
    pairs=[]
    for i in range(1,100):
        pairs = AssignPairs(employees,customers, Min_Dist_threshold, pairs)
        pairs=UpdateFrames(pairs,min_time_threshold=30,n_frames=5)
        
    print(pairs)
    
    