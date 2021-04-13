# tf-ObjectDetection

TENSORFLOW V2 ZOO Models

centernet_hg104_512x512_coco17_tpu-8
--> low fps - so takes time to change the next frame
--> detected false case in normal light - very low
--> detected true case with very low score - 0.5 or more for other objects

centernet_resnet50_v2_512x512_coco17_tpu-8
--> low fps - so takes time to change the next frame
--> detected false case in normal light - frequently
--> detected true case with very low score - 0.5 or more for other objects
--> detected all the objects tested - except bottle and book
--> detected person correctly

centernet_mobilenetv2fpn_512x512_coco17_od
⇒ could not load model

efficientdet_d0_coco17_tpu-32
--> low fps - so takes time to change the next frame lower then centernet
--> detected false case in normal light - once-twice (False for remote - it displayed cell phone)
--> detected true case with score - 0.75 around for other objects
--> detected all the objects tested - (bottle, book, cell-phone, person)

faster_rcnn_resnet50_v1_640x640_coco17_tpu-8
--> very low fps - takes more than half second to change the frame
--> detects false case - very frequently
--> detected true for cell phone and book and bottle
--> detected all objects with score more than 0.75
--> detects multiple objects for single objects - for a single cell phone in the screen shows 2 or more boxes with label cell phone.

faster_rcnn_resnet101_v1_640x640_coco17_tpu-8
--> very low fps - takes more than half second to change the frame
--> detects false case - very frequently
--> detected true for cell phone and book and bottle
--> detected all objects with score more than 0.75
--> detects multiple objects for single objects - for a single cell phone in the screen shows 2 or more boxes with label cell phone.

faster_rcnn_resnet152_v1_640x640_coco17_tpu-8
--> very low fps - got error memory excedded the usage 10%

faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8
--> very low fps
--> detected true objects
--> detected all objects with score more than 0.8
--> detects multiple objects for single objects - never
--> takes time to detect but very accurate

ssd_mobilenet_v2_320x320_coco17_tpu-8
--> good fps - detects objects fast compared to other models
--> detects false case - very frequently
--> detected true objects sometimes
--> detected all objects with score more than 0.75
--> detects multiple objects for single objects

ssd_mobilenet_v2_320x320_coco17_tpu-8
--> good fps - detects objects fast compared to other models
--> detected true objects sometimes
--> cannot detect objects all the time - not detected book, glass at all
--> detected all objects with score more than 0.75
--> detects multiple objects for single objects

ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
--> low fps
--> detected true objects
--> detected all objects with score more than 0.5
--> detects multiple objects for single objects - never
--> takes time to detect but very accurate

ssd_resnet101_v1_fpn_640x640_coco17_tpu-8
--> lower fps than resnet50
--> detected true objects
--> detected all objects with score more than 0.5
--> detects multiple objects for single objects - never
--> takes time to detect but very accurate

ssd_resnet152_v1_fpn_640x640_coco17_tpu-8
--> lower fps than resnet101
--> detected true objects
--> detected all objects with score more than 0.5
--> detects multiple objects for single objects - never
--> takes time to detect but very accurate

mask_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8
--> Allocation of 33554432 exceeds 10% of free system memory

Extremenet
→ No Proper Format
