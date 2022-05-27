from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

""" Helper function to compute IoU scores for one result """
def IoU(GROUND_TRUTH, ACTUAL, BATCH_SIZE, IMG_SIZE):
    # get predicted mask
    mask = np.argmax(ACTUAL, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    
    # get the ground truth mask into array
    ground_truth = np.zeros((BATCH_SIZE,) + IMG_SIZE + (1,), dtype="uint8")
    truth_mask = load_img(GROUND_TRUTH, target_size=IMG_SIZE, color_mode="grayscale")
    ground_truth = np.expand_dims(truth_mask, 2)
    ground_truth -=1
    
    # compute mean Intersection-Over-Union
    IOU_Keras = MeanIoU(num_classes=3) # Hardcoded for this problem: 3 classes
    IOU_Keras.update_state(ground_truth,mask)
    print("Mean IoU = ", "%.4f" % IOU_Keras.result().numpy())
    
    # print IoU per class (except background)
    values = np.array(IOU_Keras.get_weights()).reshape(3, 3)
    class1_IoU = values[1,1]/(values[1,1] + values[1,2] + values[2,1])
    class2_IoU = values[2,2]/(values[2,2] + values[2,1] + values[1,2])
    print("Class 1 IoU = ", "%.4f" % class1_IoU)
    print("Class 2 IoU = ", "%.4f" % class2_IoU)


""" Helper function to compute IoU scores for all results """
def Total_IoU(GROUND_TRUTHS, PREDICTIONS, BATCH_SIZE, IMG_SIZE):
    mean_IoU = []
    average_class1 = []
    average_class2 = []
    
    for i in range(len(PREDICTIONS)):
        # get predicted mask
        mask = np.argmax(PREDICTIONS[i], axis=-1)
        mask = np.expand_dims(mask, axis=-1)

        # get the ground truth mask into array
        ground_truth = np.zeros((BATCH_SIZE,) + IMG_SIZE + (1,), dtype="uint8")
        truth_mask = load_img(GROUND_TRUTHS[i], target_size=IMG_SIZE, color_mode="grayscale")
        ground_truth = np.expand_dims(truth_mask, 2)
        ground_truth -=1

        # compute mean Intersection-Over-Union
        IOU_Keras = MeanIoU(num_classes=3) # Hardcoded for this problem: 3 classes
        IOU_Keras.update_state(ground_truth,mask)
        mean_IoU.append(IOU_Keras.result().numpy())

        # print IoU per class (except background)
        values = np.array(IOU_Keras.get_weights()).reshape(3, 3)
        class1_IoU = values[1,1]/(values[1,1] + values[1,2] + values[2,1])
        class2_IoU = values[2,2]/(values[2,2] + values[2,1] + values[1,2])
        average_class1.append(class1_IoU)
        average_class2.append(class2_IoU)
        
    print("Overall Mean IoU = ", "%.4f" % (sum(mean_IoU)/len(mean_IoU)*100), "%")
    print("Average Class 1 IoU = ", "%.4f" % (sum(average_class1)/len(average_class1)*100), "%")
    print("Average Class 2 IoU = ", "%.4f" % (sum(average_class2)/len(average_class2)*100), "%")


""" Used to calculate the Dice Coeficient for a single prediction """
def Dice_Coef(GROUND_TRUTH, ACTUAL, BATCH_SIZE, IMG_SIZE):
    
    # get predicted mask
    mask = np.argmax(ACTUAL, axis=-1)
    mask = np.expand_dims(mask, axis=-1)

    # get the ground truth mask into array
    ground_truth = np.zeros((BATCH_SIZE,) + IMG_SIZE + (1,), dtype="uint8")
    truth_mask = load_img(GROUND_TRUTH, target_size=IMG_SIZE, color_mode="grayscale")
    ground_truth = np.expand_dims(truth_mask, 2)
    ground_truth -=1
    
    y_true_f = ground_truth.flatten()
    y_pred_f = mask.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union==0: return 1
    intersection = np.sum(y_true_f * y_pred_f)
    print("Dice Coeficient = ", "%.4f" % (intersection / union*100), "%")
