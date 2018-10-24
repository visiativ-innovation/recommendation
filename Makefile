CC = g++
CFLAGS = -Wall -O3 -fopenmp -Igzstream -Isrc -Isrc/models -std=c++0x
LDFLAGS = -lgomp -lgzstream -lz -lstdc++ -Lgzstream
OBJECTS = obj/common.o obj/corpus.o obj/model.o gzstream/gzstream.o
MODELOBJECTS = obj/models/REBUS.o obj/models/Fossil.o obj/models/FossilSimple.o obj/models/TransRec.o obj/models/TransRec_L1.o obj/models/PRME.o obj/models/HRM_max.o obj/models/HRM_avg.o obj/models/MC.o obj/models/FPMC.o obj/models/BPRMF.o obj/models/MostPopular.o

all: train

obj/model.o: src/model.hpp src/model.cpp obj/corpus.o obj/common.o  Makefile
	$(CC) $(CFLAGS) -c src/model.cpp -o $@

obj/models/MostPopular.o: src/models/MostPopular.cpp src/models/MostPopular.hpp obj/model.o obj/corpus.o obj/common.o  Makefile
	$(CC) $(CFLAGS) -c src/models/MostPopular.cpp -o $@

obj/models/BPRMF.o: src/models/BPRMF.cpp src/models/BPRMF.hpp obj/model.o obj/corpus.o obj/common.o  Makefile
	$(CC) $(CFLAGS) -c src/models/BPRMF.cpp -o $@

obj/models/MC.o: src/models/MC.cpp src/models/MC.hpp obj/model.o obj/corpus.o obj/common.o  Makefile
	$(CC) $(CFLAGS) -c src/models/MC.cpp -o $@

obj/models/Fossil.o: src/models/Fossil.cpp src/models/Fossil.hpp obj/model.o obj/corpus.o obj/common.o  Makefile
	$(CC) $(CFLAGS) -c src/models/Fossil.cpp -o $@

obj/models/FossilSimple.o: src/models/FossilSimple.cpp src/models/FossilSimple.hpp obj/model.o obj/corpus.o obj/common.o  Makefile
	$(CC) $(CFLAGS) -c src/models/FossilSimple.cpp -o $@

obj/models/FPMC.o: src/models/FPMC.cpp src/models/FPMC.hpp obj/model.o obj/corpus.o obj/common.o  Makefile
	$(CC) $(CFLAGS) -c src/models/FPMC.cpp -o $@

obj/models/HRM_max.o: src/models/HRM_max.cpp src/models/HRM_max.hpp obj/model.o obj/corpus.o obj/common.o  Makefile
	$(CC) $(CFLAGS) -c src/models/HRM_max.cpp -o $@

obj/models/HRM_avg.o: src/models/HRM_avg.cpp src/models/HRM_avg.hpp obj/model.o obj/corpus.o obj/common.o  Makefile
	$(CC) $(CFLAGS) -c src/models/HRM_avg.cpp -o $@

obj/models/PRME.o: src/models/PRME.cpp src/models/PRME.hpp obj/model.o obj/corpus.o obj/common.o  Makefile
	$(CC) $(CFLAGS) -c src/models/PRME.cpp -o $@

obj/models/TransRec.o: src/models/TransRec.cpp src/models/TransRec.hpp obj/model.o obj/corpus.o obj/common.o  Makefile
	$(CC) $(CFLAGS) -c src/models/TransRec.cpp -o $@

obj/models/TransRec_L1.o: src/models/TransRec_L1.cpp src/models/TransRec_L1.hpp obj/model.o obj/corpus.o obj/common.o  Makefile
	$(CC) $(CFLAGS) -c src/models/TransRec_L1.cpp -o $@

obj/models/REBUS.o: src/models/REBUS.cpp src/models/REBUS.hpp obj/model.o obj/corpus.o obj/common.o  Makefile
	$(CC) $(CFLAGS) -c src/models/REBUS.cpp -o $@


gzstream/gzstream.o:
	cd gzstream && make

obj/common.o: src/common.hpp src/common.cpp Makefile
	$(CC) $(CFLAGS) -c src/common.cpp -o $@

obj/corpus.o: src/corpus.hpp src/corpus.cpp obj/common.o gzstream/gzstream.o Makefile
	$(CC) $(CFLAGS) -c src/corpus.cpp -o $@

train: src/main.cpp $(OBJECTS) $(MODELOBJECTS) Makefile
	$(CC) $(CFLAGS) -o train src/main.cpp $(OBJECTS) $(MODELOBJECTS) $(LDFLAGS)

clean:
	rm -rf $(OBJECTS) $(MODELOBJECTS) train
