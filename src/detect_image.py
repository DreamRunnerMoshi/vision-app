
from detect import *

def main():

    labelsPath="yolo_v3/coco.names"
    cfgpath="yolo_v3/yolov3.cfg"
    wpath="yolo_v3/yolov3.weights"
    Lables=get_labels(labelsPath)
    CFG=get_config(cfgpath)
    Weights=get_weights(wpath)
    nets=load_model(CFG,Weights)
    Colors=get_colors(Lables)

    image = cv2.imread("./yolo_v3/person.jpg")
    res=get_predection(image,nets,Lables,Colors)
    # # show the output image
    cv2.imshow("image", res)
    cv2.waitKey()

if __name__== "__main__":
  main()