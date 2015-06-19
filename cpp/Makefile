




SRCFOLDER = src/
BUILD_FOLDER = release/
OBJFOLDER = objects/
EXPFOLDER = $(SRCFOLDER)experiments/
FRONTENDS = $(SRCFOLDER)frontends/
MATRIXCOMPLETITIONFOLDER = $(SRCFOLDER)experiments/matrixCompletition/

include cluster.mk
include gpu.mk
include multicore.mk
include cocoa.mk



all: clean	 

clean:
	rm -rf $(BUILDFOLDER)
	rm -rf $(OBJFOLDER)
	mkdir $(OBJFOLDER)
	mkdir $(BUILDFOLDER)
