python propara/run_bert.py train data/naacl18/bert/bert_params.json -s data/Outputs/bert_all -f

for prop in isOpened isSettingsChanged isPowered isInstalled isConnected isUsed isSetup isRelatedDeviceConnected
do
	python propara/run_bert.py evaluate data/Outputs/bert_all/model.tar.gz  data/Inputs/bert_annotations__test_$prop.tsv --output-file data/Outputs/bert_all_test/test_$prop.txt --cuda-device 0
	# echo $prop
	# python propara/run_bert.py train data/naacl18/bert/bert_params_$prop.json -s data/Outputs/bert_$prop
done