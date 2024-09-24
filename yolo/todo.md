### Pascal VOC xml to Yolo format

https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format
https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5


- [x] Get Data from XML file(filename, image width, image height, xmin, xmax, ymin, ymax)
- [x] Calculate box centers(x_center = xmin+(xmax-xmin)/2, y_center = ymin+(ymax-ymin)/2)
- [x] Normalize box centers(xcenter/image_width, ycenter/image_height)
- [x] Get box width and height
- [x] Normalize box width and height(width/image_width, height/image_height)
- [x] Write data to text file(same name as image file). Format: class# x_center y_center box_width box_height. One row per object.
- [x] Do for all files
- [x] Test files for correct annotations(write box on images)
- [x] Create train test folders
- [x] Fill folders: 80% train, 20% test
- [x] Create YAML file for yolo