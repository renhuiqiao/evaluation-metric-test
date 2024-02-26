import matplotlib.pyplot as plt 
import cv2
import numpy as np
import random

def generate_random_box(frame_width, frame_height, min_wh, max_wh):
    box_width = random.randint(min_wh, max_wh)
    box_height = random.randint(min_wh, max_wh)
    x1 = random.randint(0, frame_width - box_width)
    y1 = random.randint(0, frame_height - box_height)
    return (x1, y1, x1 + box_width, y1 + box_height)

def generate_random_gt(frame_width, frame_height, gt_max_wh):
    (x1, y1, x2, y2) = generate_random_box(frame_width, frame_height, 0, gt_max_wh)
    return (x1, y1, x2, y2)

def calculate_iou(box1, box2):
    # Efficient IOU calculation from
    x1_top, y1_top, x2_top, y2_top = box1
    x1_bottom, y1_bottom, x2_bottom, y2_bottom = box2

    # Determine the coordinates of the intersection rectangle
    x_left = max(x1_top, x1_bottom)
    y_top = max(y1_top, y1_bottom)
    x_right = min(x2_top, x2_bottom)
    y_bottom = min(y2_top, y2_bottom)

    # If the boxes don't intersect, return zero
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the union area
    box1_area = (x2_top - x1_top) * (y2_top - y1_top)
    box2_area = (x2_bottom - x1_bottom) * (y2_bottom - y1_bottom)

    # Calculate the IOU
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou

def inter_computation(prediction, recall):
    #walk through recall with current_item, find start and current segmentation with same recall value
    #calculate the maximum of prediction, add in to inter_prediction
    print("prediction:", prediction)
    print("recall:", recall)
    start=0
    current_item=recall[0]
    prediction_in=[]
    for j in range(len(recall)):
        current_item=recall[start]
        if recall[j]!=current_item or j==(len(recall)-1): # when the recall value changes or to the list end
            mid_prediction=prediction[start:j]
            max_prediction=np.max(mid_prediction)
            sub_list=[max_prediction]*len(mid_prediction)
            prediction_in=prediction_in+sub_list
            start=j
    
    
    print("prediction_in",prediction_in)
    return prediction_in


def compute_map(pred_inter, recall):
    #compute overall ap
    #compute map based on predition_recall curve. x-axis: recall, y-axis: precision_inter
    #along lin_cut, find corresponding precision_inter value
    #add into average_list, average the list
    lin_cut=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    average_list=[]
    current_point=0 # recall level
    for i in range(11):
        recall_level=recall[current_point]
        if lin_cut[i]<=recall_level:
            average_list.append(pred_inter[current_point])
        else:
            while(recall[current_point]==recall_level and current_point<(len(recall)-2)):
                current_point+=1    # stop at a new recall_level
            print("current_point:",current_point)
            print("len of pred_inter:", len(pred_inter))
            average_list.append(pred_inter[current_point])
    print("lin_cut:", lin_cut)
    print("average_list:", average_list)
    print("recall_len:", len(recall), "pred_len:", len(pred_inter))
    plt.plot(lin_cut, average_list, "-"); plt.show()

    Map=np.average(average_list)
    return Map

        



def VMAP(TP, FP, P_exsit, T_exist, number):
    prediction=[]
    recall=[]
    truth_sum=np.sum(T_exist)
    true_positive=0
    for i in range(number):
        true_positive+=TP[i]
        pred=true_positive*1.00/(i+1)
        prediction.append(pred)
        reca=true_positive/truth_sum
        recall.append(reca)
    prediction_inter=inter_computation(prediction, recall)
    AP=np.sum(prediction_inter)/truth_sum
    MAP=compute_map(prediction_inter, recall)
    return AP, MAP



def main():
    input_video="a_original.avi"
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("frame_width:", frame_width, "frame_height:", frame_height)
    
    box_max_wh=int(frame_height/2)
    box_min_wh=int(frame_height/2)
    num_boxes=1
    iou_threshold=0.5
    frame_no=0
    frame_record_TP=[]
    frame_record_FP=[]
    predict=[]
    truth=[]
    iou_list=[]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
                # Generate random boxes and GTs for the current frame
        boxes = generate_random_box(frame_width, frame_height, box_min_wh, box_max_wh) 
        gts = generate_random_gt(frame_width, frame_height, box_max_wh)

        print("boxes:",boxes)
        print("gts:", gts)

        # Calculate IOUs
        # TP: truth positive; FP: false positive; 
        # predict: if there is a prediction; truth: if there is truth label
        ious = calculate_iou(boxes, gts)
        iou_list.append(ious)
        print("ious:",ious)
        if ious> iou_threshold:
            frame_record_TP.append(1)
            frame_record_FP.append(0)
        else:
            frame_record_FP.append(1)
            frame_record_TP.append(0)
        predict.append(1)
        truth.append(1)
        frame_no+=1

        # (image, start_point, end_point, color, thickness)  
        frame = cv2.rectangle(frame, (boxes[0],boxes[1]), (boxes[2],boxes[3]), (255,0,0), 2) 
        frame = cv2.rectangle(frame, (gts[0],gts[1]), (gts[2],gts[3]), (0,0,255), 2) 

        cv2.imshow('Annotated Video', frame)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Release the capture and any allocated resources
    cap.release()
    cv2.destroyAllWindows()

    AP,vmap=VMAP(frame_record_TP, frame_record_FP, predict, truth, frame_no)
    average_IOU=np.average(iou_list)
    print("AP:",AP,"   Vmap:", vmap,  "    average_IOU:", average_IOU)





if __name__ == '__main__':
    main()
