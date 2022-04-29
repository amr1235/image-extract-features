import cv2
import numpy as np
def match(alg, threshold,kp1, image1,descriptors_1,image2, kp2, descriptors_2):
    mat = []
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    for i in range(len(descriptors_1)):
        for j in range(len(descriptors_2)):
            if alg =="NCC":
                out1_norm = (descriptors_1[i] - np.mean(descriptors_1[i])) / (np.std(descriptors_1[i]))
                out2_norm = (descriptors_2[j] - np.mean(descriptors_2[j])) / (np.std(descriptors_2[j]))
                # Apply similarity product between the 2 normalized vectors
                corr_vector = np.multiply(out1_norm, out2_norm)
                # Get mean of the result vector
                corr = float(np.mean(corr_vector))
                if corr >threshold:
                    mat.append([i,j,corr])
            elif alg == "SSD":
                if np.square(descriptors_1[i] - descriptors_2[j]).sum() <threshold:
                    mat.append([i,j,np.square(descriptors_1[i] - descriptors_2[j]).sum()])
    fin =[]
    for i in range(len(mat)):
        dis = np.linalg.norm(np.array(kp1[mat[i][0]].pt) - np.array(kp2[mat[i][1]].pt))
        fin.append(cv2.DMatch(mat[i][0],mat[i][1],dis))
    img3 = cv2.drawMatches(image1,kp1,image2,kp2,fin,image2,flags=2)
    return img3 