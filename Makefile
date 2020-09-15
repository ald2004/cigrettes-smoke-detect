GPU=1
ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]
OBJDIR=./obj/
DEPS = $(wildcard *.h) Makefile darknet.h
CC=gcc
NVCC=nvcc
APPNAME=app
LIBNAMESO=libdarknet.so
CPP=g++ -std=c++11
CFLAGS=-Wall -Wfatal-errors -Wno-unused-result -Wno-unknown-pragmas -Wunused-but-set-variable -Wunused-variable -Wsign-compare -fPIC
LDFLAGS= -lm -pthread
COMMON= -Iinclude/ -I3rdparty/stb/include
COMMON+= -DGPU -I/usr/local/cuda/include/
COMMON+= -DOPENCV
COMMON+= `pkg-config --cflags opencv4 2> /dev/null || pkg-config --cflags opencv`
LDFLAGS+= `pkg-config --libs opencv4 2> /dev/null || pkg-config --libs opencv`
CFLAGS+= -DGPU
CFLAGS+= -DOPENCV
CFLAGS+= -DCUDNN -I/usr/local/cudnn/include
CFLAGS+= -fPIC
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
LDFLAGS+= -lgomp
LDFLAGS+= -L/usr/local/cudnn/lib64 -lcudnn
LDFLAGS+= -lstdc++
VPATH=./src/
OBJ=image_opencv.o http_stream.o gemm.o utils.o dark_cuda.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o \
    dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o darknet.o detection_layer.o \
	captcha.o route_layer.o writing.o box.o nightmare.o normalization_layer.o avgpool_layer.o coco.o dice.o yolo.o detector.o layer.o compare.o classifier.o \
	local_layer.o swag.o shortcut_layer.o activation_layer.o rnn_layer.o gru_layer.o rnn.o rnn_vid.o crnn_layer.o demo.o tag.o cifar.o go.o batchnorm_layer.o \
	art.o region_layer.o reorg_layer.o reorg_old_layer.o super.o voxel.o tree.o yolo_layer.o gaussian_yolo_layer.o upsample_layer.o lstm_layer.o \
	conv_lstm_layer.o scale_channels_layer.o sam_layer.o
OBJ+= convolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o \
      maxpool_layer_kernels.o network_kernels.o avgpool_layer_kernels.o
OBJS = $(addprefix $(OBJDIR), $(OBJ))

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) -std=c++11 $(COMMON) $(CFLAGS) -c $< -o $@
$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@


$(OBJDIR):
	mkdir -p $(OBJDIR)

$(LIBNAMESO): $(OBJDIR) $(OBJS)
	$(CPP) -shared -std=c++11 -fvisibility=hidden -DLIB_EXPORTS $(COMMON) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

$(APPNAME): $(LIBNAMESO) $(OBJDIR) $(OBJS) 
	$(CPP) -std=c++11 $(COMMON) $(CFLAGS) $(OBJS) -o $@ src/app.c $(LDFLAGS) -L ./ -l:$(LIBNAMESO)

clean:
	rm -rf $(OBJS) ./app $(EXEC) $(LIBNAMESO) $(APPNAMESO) 
