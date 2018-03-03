sudo -H pip install pandas

cd ../
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd object_detection/
python2 choose_test_imgs_random.py
python2 xml_to_csv.py
python2 generate_tfrecord.py --images_path=images/train --csv_input=data/train_labels.csv  --output_path=data/train.record
python2 generate_tfrecord.py --images_path=images/test --csv_input=data/test_labels.csv  --output_path=data/test.record

python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config
