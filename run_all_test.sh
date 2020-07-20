python propara/run_bert.py train data/naacl18/bert/bert_params_isOpened.json -s data/Outputs/bert_isOpened -f

for folder in Scanners Printers OS Mac Linux Windows Webcams Ubuntu
do
	python propara/run_bert.py evaluate data/Outputs/bert_isOpened/model.tar.gz  data/Inputs/bert_annotations_isOpened_test_$folder.tsv --output-file data/Outputs/bert_isOpened_test/test_$folder.txt --cuda-device 0
	# echo $prop
	# python propara/run_bert.py train data/naacl18/bert/bert_params_$prop.json -s data/Outputs/bert_$prop
done