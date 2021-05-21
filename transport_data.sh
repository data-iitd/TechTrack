
if [[ $1 -le 0 ]]
then
	echo "Invalid setup number: $1. Choose from [1,2,3]."
	exit 1

elif [[ $1 -eq "1" ]] 
then
	if [[ $2  = "bert" ]]
	then
		#echo "bert"
		cp dataset/setup_1/bert/*.tsv data/Inputs/
	elif [[ $2 = "prolocal" ]]
	then
		#echo "prolocal"
		cp dataset/setup_1/prolocal/*.tsv data/Inputs/
	else
		echo "Invalid model '$2' for setup $1. Choose from [bert,prolocal]."
		exit 1
	fi


elif [[ $1 -eq "2" ]]
then
	if [[ $2  = "bert" ]]
	then
		#echo "bert"
		cp dataset/setup_2/bert/*.tsv data/Inputs/
	else
		echo "Invalid model '$2' for setup $1. Choose from [bert]."
		exit 1
	fi

elif [[ $1 -eq "3" ]]
then
	if [[ $2  = "bert" ]]
	then
		#echo "bert"
		cp dataset/setup_3/bert/*.tsv data/Inputs/
	else
		echo "Invalid model '$2' for setup $1. Choose from [bert]."
		exit 1
	fi

else
	echo "Invalid setup number $1. Choose from [1,2,3]."
	exit 1
fi

echo "[INFO] Data files setup for $2 model for setup $1 complete."
