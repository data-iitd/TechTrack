for prop in isPowered isConnected isSettingsChanged isUsed isRelatedDeviceConnected isSetup isInstalled isOpened
do
	echo $prop
	python propara/run_bert.py train data/naacl18/bert/bert_params_$prop.json -s data/Outputs/bert_$prop -f
done
python propara/run_bert.py train data/naacl18/bert/bert_params.json -s data/Outputs/bert_all -f