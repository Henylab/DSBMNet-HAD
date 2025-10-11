import numpy as np

def calculate_auc(target, predicted):
    """
    calculate AUC and false alarm auc

    Input:
        target.shape = [n,1]
        predicted.shape = [n,1]

    Output:
        PD_PF_auc
    """
    target = ((target - target.min()) /
              (target.max() - target.min()))
    predicted = ((predicted - predicted.min()) /
                 (predicted.max() - predicted.min()))
    anomaly_map = target
    normal_map = 1 - target
    num = target.shape[0]
    idx = np.argsort(predicted, axis=0)
    taus = predicted[idx].reshape((-1, 1))
    PF = np.zeros([num, 1])
    PD = np.zeros([num, 1])
    for index in range(num):
        tau = taus[index]
        anomaly_map_1 = np.double(predicted >= tau)
        PF[index] = np.sum(anomaly_map_1 * normal_map) / np.sum(normal_map)
        PD[index] = np.sum(anomaly_map_1 * anomaly_map) / np.sum(anomaly_map)
    PD_PF_auc = np.sum((PF[0:num - 1, :] - PF[1:num, :]) * (PD[1:num] + PD[0:num - 1]) / 2)
    return PD_PF_auc
def adaptive_threshold_np(image, block_size, offset):
    padded_image = np.pad(image, pad_width=block_size//2, mode='edge')
    binary_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            block = padded_image[i:i + block_size, j:j + block_size]
            local_thresh = np.mean(block) - offset
            binary_image[i, j] = 1 if image[i, j] > local_thresh else 0

    return binary_image

