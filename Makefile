install:
	# Download and install core library python dependencies
	git clone https://github.com/nicolay-r/sentiment-erc-core core
	cd core && git checkout dialog_2018
	git clone https://github.com/nicolay-r/RuSentRel data
	cd data && git checkout v1.0
	pip install -r core/dependencies.txt

download_embedding:
	wget http://rusvectores.org/static/models/rusvectores2/news_mystem_skipgram_1000_20_2015.bin.gz
	mv news_mystem_skipgram_1000_20_2015.bin.gz > data/w2v_model.bin.gz