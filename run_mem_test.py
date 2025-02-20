from remembr.planners.remember_video_planner import ReMEmbRVideoPlanner
from remembr.memloaders.memloaders import *
import json

if __name__ == "__main__":
    inpaths = [
        "/home/tiejean/RobotMem/data/captions/cobot/2025-02-08-16-39-19_VILA1.5-8b_3_secs.json",
    ]
    t_offset = 1738952666.5530548-len(inpaths)*86400 + 86400; p_offset = 0
    for i, inpath in enumerate(inpaths):
        memory = remember_cobot_vid(inpath, time_offset=t_offset, reset=(i == 0))
        t_offset += 86400
    # res = memory.video_search_by_text("cup")
    records = []
    res = memory.search_k_lastest(k=10)
    records.append(eval(res))
    res = memory.search_next_k_latest(k=10)
    records.append(eval(res))
    # res = memory.search_by_datetime("2025-02-07 12:00:00")
    import pdb; pdb.set_trace()
    # import numpy as np
    # from kneed import KneeLocator

    # # Given sorted distance scores
    # data = np.array([295.5826416015625, 301.3070068359375, 308.83160400390625, 310.0904541015625, 
    #                 313.10302734375, 314.3990478515625, 315.1335144042969, 315.5642395019531, 
    #                 315.5905456542969, 316.90704345703125, 319.6078796386719, 322.36737060546875, 
    #                 325.835205078125, 326.15673828125, 327.6396789550781, 329.3775634765625, 
    #                 329.4962463378906, 329.6998291015625, 334.1036682128906, 337.0875244140625, 
    #                 338.943115234375, 339.1634521484375, 339.18212890625, 339.9239501953125, 
    #                 341.16363525390625, 341.1875305175781, 342.4418029785156, 343.5849609375, 
    #                 343.8544006347656, 345.814453125, 345.828857421875, 348.8527526855469, 
    #                 349.677001953125, 350.88714599609375, 351.4971923828125, 352.2202453613281, 
    #                 353.2787780761719, 354.0565185546875, 354.2667236328125, 356.29522705078125, 
    #                 358.42828369140625, 358.8140869140625, 364.22747802734375, 365.5782470703125, 
    #                 367.01580810546875, 368.2586669921875, 371.391357421875, 375.4233703613281, 
    #                 381.91351318359375, 383.077392578125])
    
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(len(data)), data, marker='o', linestyle='-', label="Distance Scores")
    # plt.xlabel("Index")
    # plt.ylabel("Distance Score")
    # plt.title("Elbow Method for Cutoff (Small Values)")
    # plt.savefig("test.png")

    # # Use KneeLocator to find the elbow point
    # knee_locator = KneeLocator(range(len(data)), data, curve="concave", direction="increasing")

    # # Get the elbow index and corresponding value
    # elbow_index = knee_locator.knee
    # elbow_value = data[elbow_index] if elbow_index is not None else None

    # print(f"Elbow index: {elbow_index}")
    # print(f"Elbow value (suggested cutoff): {elbow_value}")