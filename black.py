import cv2
import numpy as np
import math
import sys
import time
import RPi.GPIO as GPIO

GPIO.setwarnings(False)

#throttle 모터 핀 - 속도제어 (후륜 2개)
throttlePin = 2 # Physical pin 3
in3 = 3 # physical Pin 5
in4 = 4 # physical Pin 7

#Steering 모터 핀 - 조향제어 (전륜 2개)
steeringPin = 22 # Physical Pin 15
in1 = 17 # Physical Pin 11
in2 = 27 # Physical Pin 13


GPIO.setmode(GPIO.BCM)		#물리적 넘버링 대신 GPIO 넘버링 사용
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)

GPIO.setup(throttlePin,GPIO.OUT)
GPIO.setup(steeringPin,GPIO.OUT)

# Steering
# in1 = 1 and in2 = 0 -> Left
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
steering = GPIO.PWM(steeringPin,1000)
steering.stop()

# Throttle
# in3 = 1 and in4 = 0 -> Forward
GPIO.output(in3,GPIO.HIGH)
GPIO.output(in4,GPIO.LOW)
throttle = GPIO.PWM(throttlePin,1000)
throttle.stop()



def detect_edges(frame):
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow("HSV",hsv)
    lower_black = np.array([0, 0, 0], dtype = "uint8")		#검정색의 하한값
    upper_black = np.array([50, 50, 100], dtype="uint8")		#검정색의 상한값
    mask = cv2.inRange(hsv,lower_black,upper_black)		#검정색 이외의 색을 걸러내서 mask에 할당
    #cv2.imshow("mask",mask)	
    
    # detect edges
    edges = cv2.Canny(mask, 50, 100)
    #cv2.imshow("edges",edges)
    
    return edges

def region_of_interest(edges):
    height, width = edges.shape		#엣지 프레임의 높이와 너비를 추출
    mask = np.zeros_like(edges)		#엣지 프레임의 동일한 차원으로 빈 행렬을 만들어 mask에 할당


    polygon = np.array([[			#화면의 아래쪽 절반에만 초점을 맞춤
        (0, height),				#4점의 좌표를 지정(좌하,좌상,우상,우하)
        (0,  height/2),
        (width , height/2),
        (width , height),
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)		#다각형을 파란색으로 채우라는 의미
    
    cropped_edges = cv2.bitwise_and(edges, mask)
    cv2.imshow("roi",cropped_edges)
    
    return cropped_edges

def detect_line_segments(cropped_edges):
    rho = 1  
    theta = np.pi / 180  
    min_threshold = 10  
    
    line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold,		#rho = 픽셀 단위의 거리 정밀도(일반적으로 1로 설정)
                                    np.array([]), minLineLength=5, maxLineGap=150)			#theta = 라디안 단위의 각도 정밀도(항상 np.pi/180 ~ 1도로 설정)
										#min_threshold = 라인으로 인식하게끔하기 위해 얻어야 하는 최수 수
    return line_segments								#minLineLength = 라인의 최소 길이(픽셀). 이 보다 짧은 줄은 줄로 인식하지 않음
										#maxLineGap = 라인 2개 사이의 최대 픽셀 간격	
def average_slope_intercept(frame, line_segments):
    lane_lines = []
    
    if line_segments is None:
        print("no line segments detected")
        return lane_lines

    height, width,_ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary
    
    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                print("skipping vertical lines (slope = infinity")
                continue
            
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    return lane_lines

def make_points(frame, line):		#make_points() = 레인 라인의 경계 좌표를 반환하는 average_slope_intercept()함수의 도우미 함수
    height, width, _ = frame.shape
    
    slope, intercept = line
    
    y1 = height		#프레임 하단
    y2 = int(y1 / 2)		#프레임 중앙에서 아래로 점 만듦
    
    if slope == 0:
        slope = 0.1
        
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return [[x1, y1, x2, y2]]

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6):		#선 색상(B,G,R)
    line_image = np.zeros_like(frame)
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
                
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)		#cv2.addWeighted() = 괄호 안의 매개변수를 사용하며 두 이미지를 가중치를 부여하여 결합하는 데 사용
    
    return line_image


def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5 ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    
    steering_angle_radian = steering_angle / 180.0 * math.pi
    
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)
    
    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    
    return heading_image

def get_steering_angle(frame, lane_lines):
    
    height,width,_ = frame.shape
    
    if len(lane_lines) == 2:		#두 개의 레인 라인이 감지
        _, _, left_x2, _ = lane_lines[0][0]		#Lane_lines 배열에서 왼쪽 x2 추출
        _, _, right_x2, _ = lane_lines[1][0]		#Lane_lines 배열에서 오른쪽 x2 추출
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid		#평균이 화면 중앙과 얼마나 다른지 나타낸다
        y_offset = int(height / 2)			#항상 동일
        
    elif len(lane_lines) == 1:		#한 개의 레인 라인이 감지
        x1, _, x2, _ = lane_lines[0][0]		
        x_offset = x2 - x1
        y_offset = int(height / 2)
        
    elif len(lane_lines) == 0:		#레인 라인이 감지되지 않는 경우
        x_offset = 0
        y_offset = int(height / 2)
        
    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  
    steering_angle = angle_to_mid_deg + 90		#'steering_angle = 90'이면 자동차가 수직방향선을 가지며 조향값 없이 전진한다는 의미 
    						#'steering_angle > 90'이면 오른쪽으로 조종해야 하고 90보다 작을 경우에는 왼쪽으로 조종해야 한다
    return steering_angle

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH,320)		#너비를 320p로 설정
video.set(cv2.CAP_PROP_FRAME_HEIGHT,240)		#높이를 240p로 설정

time.sleep(1)

##fourcc = cv2.VideoWriter_fourcc(*'XVID')
##out = cv2.VideoWriter('Original15.avi',fourcc,10,(320,240))
##out2 = cv2.VideoWriter('Direction15.avi',fourcc,10,(320,240))

#루프마다 업데이트할 변수
speed = 8
lastTime = 0
lastError = 0

#PD 상수
kp = 0.4
kd = kp * 0.65

while True:
    ret,frame = video.read()		#카메라에서 가져온 각 프레임을 읽고 frame변수에 저장
    frame = cv2.flip(frame,1)		#-1로 설정시 카메라 상하반전
    	#함수 호출
    cv2.imshow("original",frame)
    edges = detect_edges(frame)
    roi = region_of_interest(edges)
    line_segments = detect_line_segments(roi)
    lane_lines = average_slope_intercept(frame,line_segments)
    lane_lines_image = display_lines(frame,lane_lines)
    steering_angle = get_steering_angle(frame, lane_lines)
    heading_image = display_heading_line(lane_lines_image,steering_angle)
    cv2.imshow("heading line",heading_image)

    now = time.time()
    dt = now - lastTime

    deviation = steering_angle - 90		#angle_to_mid_deg 변수와 동일
    error = abs(deviation)
    
    if deviation < 5 and deviation > -5:	#10도 오류 범위가 있는 경우에는 조종하지 않음
        deviation = 0
        error = 0
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.LOW)
        steering.stop()

    elif deviation > 5:		#편차가 양수이면 우회전
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.HIGH)
        steering.start(100)
        

    elif deviation < -5:		#편차가 음수이면 왼쪽으로 조정
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        steering.start(100)

    derivative = kd * (error - lastError) / dt
    proportional = kp * error
    PD = int(speed + derivative + proportional)
    spd = abs(PD)

    if spd > 25:
        spd = 25
        
    throttle.start(spd)

    lastError = error
    lastTime = time.time()
        
##    out.write(frame)
##    out2.write(heading_image)

    key = cv2.waitKey(1)
    if key == 27:
        break
    
video.release()
##out.release()
##out2.release()
cv2.destroyAllWindows()
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
GPIO.output(in3,GPIO.LOW)
GPIO.output(in4,GPIO.LOW)
throttle.stop()
steering.stop()


