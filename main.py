import os
import numpy as np
import cv2
import typing as Text

path = "C:/Users/ERVIN/Desktop/vehicle_Speed/vehicle_tracking/speeding_vehicles"
pts = np.array([[413,360],[716,360],[740,400],[377,400]], np.int32)
pts = pts.reshape(-1,1,2)


# [[x, y, w, h, cX, cY, carID, speed, check_speed]]
car_tracking = np.array([[]])
new_tracker = 0
carID = 0



def calc_speed(curr_Pos, last_pos):
    curr_x = curr_Pos[4]
    prev_x = last_pos[4]
    curr_y = curr_Pos[5]+360
    prev_y = last_pos[5]+360

    # Pixels per 12 Feet
    pixel_per_12feet_curr_y= -1.82 + 0.146*curr_y + 0.000376*(curr_y**2)
    pixel_per_12feet_prev_y=-1.82 + 0.146*prev_y + 0.000376*(prev_y**2)

    # y-axis Pixel to feet
    curr_y= (-676.8992) + (2.7365*curr_y) + (-0.00237*(curr_y**2))
    prev_y= (-676.8992) + (2.7365*prev_y) + (-0.00237*(prev_y**2))
    
    # x-axis Pixel to feet
    curr_x = curr_x * (12 / pixel_per_12feet_curr_y)
    prev_x = prev_x * (12 / pixel_per_12feet_prev_y)
    
    # Compute distance travel and convert to kilometer
    distance = np.sqrt(((curr_x - prev_x)**2) + ((curr_y-prev_y)**2)) * 0.0003048
    # Convert to km/s to km/h
    kph = distance / (2/30) * 3600

    print("Distance: ", kph)
    curr_Pos[8] = True
    return kph





vid = cv2.VideoCapture('./cctv.mp4')
while(1):
#---------------------------------------------------------------------------------------
    # Object Detection

    # Background Subtraction
    _,frame1 = vid.read()
    roi = frame1[360:,:,:]
    grayscaled1 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    grayscaled1[grayscaled1<0] = 0
    grayscaled1[grayscaled1>255] = 255

    cv2.imshow("First Frame", frame1)

    vid.grab()

    _,frame2 = vid.read()
    roi = frame2[360:,:,:]
    grayscaled2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    grayscaled2[grayscaled2<0] = 0
    grayscaled2[grayscaled2>255] = 255


    difference = np.int16(grayscaled2)-np.int16(grayscaled1)

    difference[difference<0] = 0
    difference[difference>255] = 255

    difference[difference<50] = 0
    difference[difference>=50] = 255

    difference = np.uint8(difference)

    cv2.imshow("Difference",difference)

    
# ---------------------------------------------------------------------------------------
    # Image Processing

    mask = np.ones((5,5), np.uint8)
    
    dilate = cv2.morphologyEx(difference, cv2.MORPH_DILATE, mask, iterations=5)
    close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, mask, iterations=5)
    open = cv2.morphologyEx(close, cv2.MORPH_OPEN, mask, iterations=5)

    cv2.imshow("Masked", close)
    
# ---------------------------------------------------------------------------------------
    # Image Segmentation

    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(close, 4, cv2.CV_32S)

#----------------------------------------------------------------------------------------
    #Object Tracking

    # Append and Update Trackers
    i = 1
    while(i < numLabels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        new_tracker = np.array([[x,y,w,h,cX,cY,carID,0, False]])

        # if(w <= 160 and w >= 50 and h <= 160 and h >= 70):
        matchCarID = None

        if(car_tracking.size != 0):  
            for car in car_tracking:
                prev_x = car[0]
                prev_y = car[1]
                prev_w = car[2]
                prev_h = car[3]
                prev_cX = car[4]
                prev_cY = car[5]
                
                if((prev_x <= cX <= (prev_x+prev_w)) and (prev_y <= cY <= (prev_y+prev_h)) and (x <= prev_cX <= (x + w)) and (y <= prev_cY <= (y + h))):
                    print("Updating Trackers...")
                    matchCarID = i

                    if(car[8] != True):
                        speed = calc_speed(new_tracker[0], car)
                        car[7] = speed
                        car[8] = True

                    car[0] = x
                    car[1] = y
                    car[2] = w
                    car[3] = h
                    car[4] = cX
                    car[5] = cY
            
            if(matchCarID == None):
                print("Appending new Tracker... [ID: ", carID,"]")
                car_tracking = np.vstack((car_tracking, new_tracker))
                carID += 1
                
        else:
            print("Appending new Tracker... [ID: ", carID,"]")
            car_tracking = np.array(new_tracker)
            carID += 1


        i+=1

# -------------------------------------------------------------------------------------------------------------------------------
    # Updating Tracking Locations

    print(car_tracking)
    i=0
    # Update new points
    if(car_tracking.size != 0):
        
        # Delete Unwanted Points
        while(i < car_tracking.shape[0]):
            if(car_tracking[i][5]+360 > 500):
                print("Deleting Tracker... [ID: ", car_tracking[i][6], "]")
                car_tracking = np.delete(car_tracking, i, 0)
            i+=1

        # Update Points
        for car in car_tracking:
            curr_x = car[0]
            curr_y = car[1]
            curr_w = car[2]
            curr_h = car[3]
            curr_cX = car[4]
            curr_cY = car[5]
            curr_car_id = car[6]
            speed = car[7]
            
            if(speed > 0):
                cv2.rectangle(frame2, (int(curr_x),int(curr_y+360)),(int(curr_x+curr_w),int(curr_y+360+curr_h)), (0,255,0), 1)
                cv2.circle(frame2, (int(curr_cX),int(curr_cY+360)), 3, (0,0,255),-1)
                cv2.putText(frame2, Text.Text(int(speed)) + " kph", (int(curr_cX),int(curr_cY+360)), cv2.QT_FONT_NORMAL, 0.5, (0,255,0), 1)

                if(speed > 80 and (curr_cY+360) <= 400):
                    print("CAR_ID: ", carID, "has exceeded the limit!!!!!!!")
                    # path = "C:/Users\ERVIN\Desktop/vehicle_Speed/vehicle_tracking/speeding_cars"
                    sc = frame2[int(curr_y+360) : int(curr_y+360+curr_h), int(curr_x) : int(curr_x+curr_w), :]
                    # cv2.imshow("Speeding", sc)
                    
                    cv2.imwrite(os.path.join(path,"CarID_" + str(curr_car_id) + ".png"), sc)



    
    # cv2.line(frame2, (0,360),(1280,360),(0,255,0),1)
    # cv2.line(frame2, (0,400),(1280,400),(0,255,0),1)
    
    cv2.polylines(frame2, [pts],True,(0,255,0),1)

    cv2.imshow("Second Frame", frame2)
    # cv2.imwrite("SCREENSHOT.png", frame2)

    k = cv2.waitKey(50)
    if k == 27:
        break

vid.release()
cv2.destroyAllWindows()