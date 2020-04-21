##
## Makefile
##
## Pieter Robberechts
## Nov 2018
## 


# Experiment parameters ######################################################

SMALL_DATA=../data/dataSubset/index_small
SMALL_PERIOD=100
SMALL_OUT=small

DATA=/cw/bdap/assignment3/Data/index
PERIOD=1000
OUT=out

STOPWORDS=./stop-word-list_stanford.txt

MAX_N=4
LOG_NB_BUCKETS=15
NB_HASHES=5
THRESHOLD=0.5
LEARNING_RATE=0.01

# Compilation  ###############################################################

## Locate directories
class_d=bin
lib_d=lib
source_d=src

# Compilation stuff
JAVAC=javac
JFLAGS=-g -d $(class_d) -sourcepath $(source_d) -Xlint:all

classpath:=$(class_d):$(lib_d)/javax.mail.jar
# If there's already a CLASSPATH, put it on the front
ifneq ($(CLASSPATH),)
        classpath:= $(CLASSPATH):$(classpath)
endif
# Re-export the CLASSPATH.
export CLASSPATH:=$(classpath)

.SUFFIXES: .java .class
.PHONY: clean all

$(class_d)/%.class: $(source_d)/%.java
	@echo "JAVAC $<"
	@$(JAVAC) $(JFLAGS) $<

PROG= \
			NaiveBayesFeatureHashing.class \
			NaiveBayesCountMinSketch.class \
			PerceptronFeatureHashing.class \
			PerceptronCountMinSketch.class
LIST=$(addprefix $(class_d)/, $(PROG))
	
all: $(class_d) $(LIST) 

$(class_d):
	mkdir $(class_d)

clean:
	rm -rf $(class_d)/*

# Experiments ################################################################

nbfh_small: $(class_d)/NaiveBayesFeatureHashing.class
	@echo "Testing naive Bayes with feature hashing on a subset of the data"
	rm -f $(SMALL_OUT).nbfh.*
	time java NaiveBayesFeatureHashing $(SMALL_DATA) $(STOPWORDS) $(LOG_NB_BUCKETS) $(THRESHOLD) $(SMALL_OUT) $(SMALL_PERIOD) $(MAX_N) -writeOutAllPredictions

nbfh: $(class_d)/NaiveBayesFeatureHashing.class
	@echo "Testing naive Bayes with feature hashing on the complete data"
	rm -f $(OUT).nbfh.*
	time java NaiveBayesFeatureHashing $(DATA) $(STOPWORDS) $(LOG_NB_BUCKETS) $(THRESHOLD) $(OUT) $(PERIOD) $(MAX_N)

nbcms_small: $(class_d)/NaiveBayesCountMinSketch.class
	@echo "Testing naive Bayes with count-min sketch on a subset of the data"
	rm -f $(SMALL_OUT).nbcms.*
	time java NaiveBayesCountMinSketch $(SMALL_DATA) $(STOPWORDS) $(LOG_NB_BUCKETS) $(NB_HASHES) $(THRESHOLD) $(SMALL_OUT) $(SMALL_PERIOD) $(MAX_N) -writeOutAllPredictions

nbcms: $(class_d)/NaiveBayesCountMinSketch.class
	@echo "Testing naive Bayes with count-min sketch on the complete data"
	rm -f $(OUT).nbcms.*
	time java NaiveBayesCountMinSketch $(DATA) $(STOPWORDS) $(LOG_NB_BUCKETS) $(NB_HASHES) $(THRESHOLD) $(OUT) $(PERIOD) $(MAX_N)

pfh_small: $(class_d)/PerceptronFeatureHashing.class
	@echo "Testing perceptron classification with feature hashing on a subset of the data"
	rm -f $(SMALL_OUT).pfh.*
	time java PerceptronFeatureHashing $(SMALL_DATA) $(STOPWORDS) $(LOG_NB_BUCKETS) $(LEARNING_RATE) $(SMALL_OUT) $(SMALL_PERIOD) $(MAX_N) -writeOutAllPredictions

pfh: $(class_d)/PerceptronFeatureHashing.class
	@echo "Testing perceptron classification with feature hashing on the complete data"
	rm -f $(OUT).pfh.*
	time java PerceptronFeatureHashing $(DATA) $(STOPWORDS) $(LOG_NB_BUCKETS) $(LEARNING_RATE) $(OUT) $(PERIOD) $(MAX_N)

pcms_small: $(class_d)/PerceptronCountMinSketch.class
	@echo "Testing perceptron classification with count-min sketch on a subset of the data"
	rm -f $(SMALL_OUT).pcms.*
	time java PerceptronCountMinSketch $(SMALL_DATA) $(STOPWORDS) $(LOG_NB_BUCKETS) $(NB_HASHES) $(LEARNING_RATE) $(SMALL_OUT) $(SMALL_PERIOD) $(MAX_N) -writeOutAllPredictions

pcms: $(class_d)/PerceptronCountMinSketch.class
	@echo "Testing perceptron classification with count-min sketch on the complete data"
	rm -f $(OUT).pcms.*
	time java PerceptronCountMinSketch $(DATA) $(STOPWORDS) $(LOG_NB_BUCKETS) $(NB_HASHES) $(LEARNING_RATE) $(OUT) $(PERIOD) $(MAX_N)

adpfh_small: $(class_d)/AdjustedPerceptronFeatureHashing.class
	@echo "Testing enchanched perceptron classification with feature hashing on a subset of the data"
	rm -f $(SMALL_OUT).adpfh.*
	time java AdjustedPerceptronFeatureHashing $(SMALL_DATA) $(STOPWORDS) $(LOG_NB_BUCKETS) $(THRESHOLD) $(LEARNING_RATE) $(SMALL_OUT) $(SMALL_PERIOD) $(MAX_N) -writeOutAllPredictions

adpfh: $(class_d)/AdjustedPerceptronFeatureHashing.class
	@echo "Testing perceptron classification with count-min sketch on the complete data"
	rm -f $(OUT).adpfh.*
	time java AdjustedPerceptronFeatureHashing $(DATA) $(STOPWORDS) $(LOG_NB_BUCKETS) $(THRESHOLD) $(LEARNING_RATE) $(OUT) $(PERIOD) $(MAX_N)
