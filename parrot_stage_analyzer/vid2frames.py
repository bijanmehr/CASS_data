import numpy as np
import cv2
from progress.bar import Bar
import os
import sys



if len(sys.argv) > 2:
    print('You have specified too many arguments')
    sys.exit()

if len(sys.argv) < 2:
    print('You need to specify the path to be listed')
    sys.exit()

input_path = sys.argv[1]

if not os.path.isfile(input_path):
    print('The path specified does not exist')
    sys.exit()

try:
    os.mkdir("frames")
except OSError:
    print ("Creation of the directory %s failed" % "frames")
else:
    print ("Successfully created the directory %s " % "frames")

capture = cv2.VideoCapture(input_path)
length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
bar = Bar('Processing Frames', max=length)

i = 0
while (capture.isOpened()):
    ret, frame = capture.read()
    name = "./frames/frame%d.jpg" % i
    cv2.imwrite(name, frame)
    i = i + 1
    bar.next()
    if i == length:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

bar.finish()

temp = input_path
os.rename("frames", temp[temp.find("_")-2:temp.find("start")-1] + "_frames")

# cleanup
capture.release()
cv2.destroyAllWindows()

