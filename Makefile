install:
	# Download and install AREkit core library with python dependencies
	git clone git@github.com:nicolay-r/AREkit.git core
	# Switching to the branch with the related experiments
	cd core && git checkout 0.18.1-dialog-rc
	git clone https://github.com/nicolay-r/RuSentRel temp
	mv temp/ data/
	cd data && git checkout v1.0
	pip install -r core/dependencies.txt

download_embedding:
	wget http://rusvectores.org/static/models/rusvectores2/news_mystem_skipgram_1000_20_2015.bin.gz
	mv news_mystem_skipgram_1000_20_2015.bin.gz data/w2v_model.bin.gz
