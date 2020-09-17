#include <stdlib.h>
#include "utils.h"
#include "dark_cuda.h"
#include "image_opencv.h"
#include "option_list.h"
#include "parser.h"

#ifdef GPUX
extern "C" {void run_detector(int argc, char** argv); }
#endif // GPU

network parse_network_cfg_custx(char* filename);


int main(int argc, char** argv) {
	int i;
    int check_mistakes = 0;
	for (i = 0; i < argc; i++) {
		if (!argv[i])continue;
		strip_args(argv[i]);
	}
	if (argc < 2) {
		fprintf(stderr, "usage: %s <function>\n", argv[0]);
		return 0;
	}
	gpu_index = find_int_arg(argc, argv, "-i", 0);
	if (find_arg(argc, argv, "-nogpu")) {
		gpu_index = -1;
		printf("\n Currently Darknet doesn't support -nogpu flag. If you want to use CPU - please compile Darknet with GPU=0 in the Makefile, or compile darknet_no_gpu.sln on Windows.\n");
		exit(-1);
	}
#ifdef GPU
	if (gpu_index >= 0) {
		cuda_set_device(gpu_index);
		CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
	}
	show_cuda_cudnn_info();
#endif
	//cuda_debug_sync = find_arg(argc, argv, "-cuda_debug_sync");
#ifdef OPENCV
	show_opencv_info();
#endif

	//if (0 == strcmp(argv[1], "detector")) {
	//	run_detector(argc, argv);
	//}

    int dont_show = find_arg(argc, argv, "-dont_show");
    int benchmark = find_arg(argc, argv, "-benchmark");
    int benchmark_layers = find_arg(argc, argv, "-benchmark_layers");
    //if (benchmark_layers) benchmark = 1;
    if (benchmark) dont_show = 1;
    int show = find_arg(argc, argv, "-show");
    int letter_box = find_arg(argc, argv, "-letter_box");
    int calc_map = find_arg(argc, argv, "-map");
    int map_points = find_int_arg(argc, argv, "-points", 0);
    check_mistakes = find_arg(argc, argv, "-check_mistakes");
    int show_imgs = find_arg(argc, argv, "-show_imgs");
    int mjpeg_port = find_int_arg(argc, argv, "-mjpeg_port", -1);
    int avgframes = find_int_arg(argc, argv, "-avgframes", 3);
    int dontdraw_bbox = find_arg(argc, argv, "-dontdraw_bbox");
    int json_port = find_int_arg(argc, argv, "-json_port", -1);
    char* http_post_host = find_char_arg(argc, argv, "-http_post_host", 0);
    int time_limit_sec = find_int_arg(argc, argv, "-time_limit_sec", 0);
    char* out_filename = find_char_arg(argc, argv, "-out_filename", 0);
    char* outfile = find_char_arg(argc, argv, "-out", 0);
    char* prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .25);    // 0.24
    float iou_thresh = find_float_arg(argc, argv, "-iou_thresh", .5);    // 0.5 for mAP
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int num_of_clusters = find_int_arg(argc, argv, "-num_of_clusters", 5);
    int width = find_int_arg(argc, argv, "-width", -1);
    int height = find_int_arg(argc, argv, "-height", -1);
    // extended output in test mode (output of rect bound coords)
    // and for recall mode (extended output table-like format with results for best_class fit)
    int ext_output = find_arg(argc, argv, "-ext_output");
    int save_labels = find_arg(argc, argv, "-save_labels");
    char* chart_path = find_char_arg(argc, argv, "-chart", 0);
    if (argc < 4) {
        fprintf(stderr, "usage: %s %s [train/test/valid/demo/map] [data] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return 0;
    }
    static int coco_ids[] = { 1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90 };

    char* gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int* gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if (gpu_list){ 
        printf("the gpu list is: %s \n", gpu_list);
        int len = (int)strlen(gpu_list);
        ngpus = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = (int*)xcalloc(ngpus, sizeof(int));
        for (i = 0; i < ngpus; ++i) {
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',') + 1;
        }
    }
    else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }
    int clear = find_arg(argc, argv, "-clear");

    char* datacfg = argv[3];
    char* cfg = argv[4];
    char* weights = (argc > 5) ? argv[5] : 0;
    if (weights)
        if (strlen(weights) > 0)
            if (weights[strlen(weights) - 1] == 0x0d) weights[strlen(weights) - 1] = 0;
    char* filename = (argc > 6) ? argv[6] : 0;
    //train_detector(datacfg, cfg, weights, gpus, ngpus, clear, dont_show, calc_map, mjpeg_port, show_imgs, benchmark_layers, chart_path);
    list* options = read_data_cfg(datacfg);
    char* train_images = option_find_str(options, "train", "data/train.txt");
    char* valid_images = option_find_str(options, "valid", train_images);
    char* backup_directory = option_find_str(options, "backup", "/backup/");
    
    //{
    //    //debug 
    //    node* t = options->front;
    //    int nu = 0;
    //    while (1)
    //    {
    //        ++nu;
    //        kvp* val = t->val;
    //        if (!t || !t->next)break;
    //        printf("%s == %s \n", val->key, val->val);
    //        t = t->next;
    //    }
    //    printf("********* total cfg file lines : %d \n", nu);
    //}
    srand(time(0));
    char* base = basecfg(cfg);
    printf("%s\n", base); //smoke_small
    //exit(0);
    float avg_loss = -1;
    float avg_contrastive_acc = 0;
    network* nets = (network*)xcalloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int k;
    for (k = 0; k < ngpus; ++k) {
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[k]);
#endif

        //preallocate network memory * batch for cuda.
        //create cudaStearm.
        //create pinned memory.if memory optimize is set.
        nets[k] = parse_network_cfg_custx(cfg);

        //nets[k].benchmark_layers = benchmark_layers;
        /*if (weights) {
            load_weights(&nets[k], weights);
        }*/
        if (clear) {
            *nets[k].seen = 0;
            *nets[k].cur_iteration = 0;
        }
        nets[k].learning_rate *= ngpus;
    }

    network net = nets[0];
    const int actual_batch_size = net.batch * net.subdivisions;
    int img = net.batch * net.subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);

    //getchar();
    //end of file
    if (gpus && gpu_list && ngpus > 1) free(gpus);
}

network parse_network_cfg_custx(char* filename) {
    list* sections = read_cfg(filename);
    if (!sections->front)error("Config file has no sections");
    node* nodefirst = sections->front;
    network net = make_network(sections->size-1);
    net.gpu_index = gpu_index;
    size_params params;
    params.train = 1;
    section* s = (section *)nodefirst->val;
    list* options = s->options;
    if (!strcmp(s->type, "[net]") && !strcmp(s->type, "[network]")) {
        fprintf(stderr, "%s ", s->type);
        error("First section must be [net] or [network]");
    }
    parse_net_options(options, &net);
#ifdef GPU
    printf("net.optimized_memory = %d params.train = %d \n", net.optimized_memory, params.train);
    if (net.optimized_memory >= 2 && params.train) {
        pre_allocate_pinned_memory((size_t)1024 * 1024 * 1024 * 1);   // pre-allocate 8 GB CPU-RAM for pinned memory
    }
#endif // GPU
    params.h = net.h;
    params.w = net.w;
    params.c = net.c;
    params.inputs = net.inputs;
    //if (batch > 0) net.batch = batch;
    //if (time_steps > 0) net.time_steps = time_steps;
    if (net.batch < 1) net.batch = 1;
    if (net.time_steps < 1) net.time_steps = 1;
    if (net.batch < net.time_steps) net.batch = net.time_steps;
    params.batch = net.batch;
    params.time_steps = net.time_steps;
    params.net = net;
    printf("mini_batch = %d, batch = %d, time_steps = %d, train = %d \n", net.batch, net.batch * net.subdivisions, net.time_steps, params.train);
    int avg_outputs = 0;
    int avg_counter = 0;
    float bflops = 0;
    size_t workspace_size = 0;
    size_t max_inputs = 0;
    size_t max_outputs = 0;
    int receptive_w = 1, receptive_h = 1;
    int receptive_w_scale = 1, receptive_h_scale = 1;
    const int show_receptive_field = option_find_float_quiet(options, "show_receptive_field", 0);
    nodefirst = nodefirst->next;
    int count = 0;
    free_section(s);
    //fprintf(stderr, "   layer   filters  size/strd(dil)      input                output\n");
    /*[convolutional]
    [maxpool]
    [net]
    [route]
    [sam]
    [shortcut]
    [upsample]
    [yolo]*/
    while (nodefirst){
        params.index = count;
        //fprintf(stderr, "%4d ", count);
        s = (section*)nodefirst->val;
        options = s->options;
        layer l = { (LAYER_TYPE)0 };
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if (lt == CONVOLUTIONAL) {
            l = parse_convolutional(options, params);
        }
        else if (lt == LOCAL) {
            l = parse_local(options, params);
        }
        else if (lt == ACTIVE) {
            l = parse_activation(options, params);
        }
        else if (lt == CONNECTED) {
            l = parse_connected(options, params);
        }
        else if (lt == CROP) {
            l = parse_crop(options, params);
        }
        else if (lt == COST) {
            l = parse_cost(options, params);
            l.keep_delta_gpu = 1;
        }
        else if (lt == REGION) {
            l = parse_region(options, params);
            l.keep_delta_gpu = 1;
        }
        else if (lt == YOLO) {
            l = parse_yolo(options, params);
            l.keep_delta_gpu = 1;
        }
        else if (lt == GAUSSIAN_YOLO) {
            l = parse_gaussian_yolo(options, params);
            l.keep_delta_gpu = 1;
        }
        else if (lt == DETECTION) {
            l = parse_detection(options, params);
        }
        else if (lt == SOFTMAX) {
            l = parse_softmax(options, params);
            net.hierarchy = l.softmax_tree;
            l.keep_delta_gpu = 1;
        }
        else if (lt == NORMALIZATION) {
            l = parse_normalization(options, params);
        }
        else if (lt == BATCHNORM) {
            l = parse_batchnorm(options, params);
        }
        else if (lt == MAXPOOL) {
            l = parse_maxpool(options, params);
        }
        else if (lt == LOCAL_AVGPOOL) {
            l = parse_local_avgpool(options, params);
        }
        else if (lt == REORG) {
            l = parse_reorg(options, params);
        }
        else if (lt == REORG_OLD) {
            l = parse_reorg_old(options, params);
        }
        else if (lt == AVGPOOL) {
            l = parse_avgpool(options, params);
        }
        else if (lt == ROUTE) {
            l = parse_route(options, params);
            int k;
            for (k = 0; k < l.n; ++k) {
                net.layers[l.input_layers[k]].use_bin_output = 0;
                net.layers[l.input_layers[k]].keep_delta_gpu = 1;
            }
        }
        else if (lt == UPSAMPLE) {
            l = parse_upsample(options, params, net);
        }
        else if (lt == SHORTCUT) {
            l = parse_shortcut(options, params, net);
            net.layers[count - 1].use_bin_output = 0;
            net.layers[l.index].use_bin_output = 0;
            net.layers[l.index].keep_delta_gpu = 1;
        }
        else if (lt == SCALE_CHANNELS) {
            l = parse_scale_channels(options, params, net);
            net.layers[count - 1].use_bin_output = 0;
            net.layers[l.index].use_bin_output = 0;
            net.layers[l.index].keep_delta_gpu = 1;
        }
        else if (lt == SAM) {
            l = parse_sam(options, params, net);
            net.layers[count - 1].use_bin_output = 0;
            net.layers[l.index].use_bin_output = 0;
            net.layers[l.index].keep_delta_gpu = 1;
        }
        else if (lt == DROPOUT) {
            l = parse_dropout(options, params);
            l.output = net.layers[count - 1].output;
            l.delta = net.layers[count - 1].delta;
#ifdef GPU
            l.output_gpu = net.layers[count - 1].output_gpu;
            l.delta_gpu = net.layers[count - 1].delta_gpu;
            l.keep_delta_gpu = 1;
#endif
        }
        else if (lt == EMPTY) {
            layer empty_layer = { (LAYER_TYPE)0 };
            empty_layer.out_w = params.w;
            empty_layer.out_h = params.h;
            empty_layer.out_c = params.c;
            l = empty_layer;
            l.output = net.layers[count - 1].output;
            l.delta = net.layers[count - 1].delta;
#ifdef GPU
            l.output_gpu = net.layers[count - 1].output_gpu;
            l.delta_gpu = net.layers[count - 1].delta_gpu;
#endif
        }
        else {
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }

        
#ifdef GPU
        // futher GPU-memory optimization: net.optimized_memory == 2
        l.optimized_memory = net.optimized_memory;
        if (net.optimized_memory >= 2 && params.train && l.type != DROPOUT)
        {
            if (l.output_gpu) {
                cuda_free(l.output_gpu);
                //l.output_gpu = cuda_make_array_pinned(l.output, l.batch*l.outputs); // l.steps
                l.output_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch * l.outputs); // l.steps
            }
            if (l.activation_input_gpu) {
                cuda_free(l.activation_input_gpu);
                l.activation_input_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch * l.outputs); // l.steps
            }

            if (l.x_gpu) {
                cuda_free(l.x_gpu);
                l.x_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch * l.outputs); // l.steps
            }

            // maximum optimization
            if (net.optimized_memory >= 3 && l.type != DROPOUT) {
                if (l.delta_gpu) {
                    cuda_free(l.delta_gpu);
                    //l.delta_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
                    //printf("\n\n PINNED DELTA GPU = %d \n", l.batch*l.outputs);
                }
            }

            if (l.type == CONVOLUTIONAL) {
                set_specified_workspace_limit(&l, net.workspace_size_limit);   // workspace size limit 1 GB
            }
        }
#endif // GPU

        l.clip = option_find_float_quiet(options, "clip", 0);
        l.dynamic_minibatch = net.dynamic_minibatch;
        l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l.dont_update = option_find_int_quiet(options, "dont_update", 0);
        l.burnin_update = option_find_int_quiet(options, "burnin_update", 0);
        l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l.train_only_bn = option_find_int_quiet(options, "train_only_bn", 0);
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        option_unused(options);
        net.layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        if (l.inputs > max_inputs) max_inputs = l.inputs;
        if (l.outputs > max_outputs) max_outputs = l.outputs;
        free_section(s);
        nodefirst = nodefirst->next;
        ++count;
        if (nodefirst) {
            if (l.antialiasing) {
                params.h = l.input_layer->out_h;
                params.w = l.input_layer->out_w;
                params.c = l.input_layer->out_c;
                params.inputs = l.input_layer->outputs;
            }
            else {
                params.h = l.out_h;
                params.w = l.out_w;
                params.c = l.out_c;
                params.inputs = l.outputs;
            }
        }
        if (l.bflops > 0) bflops += l.bflops;

        if (l.w > 1 && l.h > 1) {
            avg_outputs += l.outputs;
            avg_counter++;
        }
    }
    free_list(sections);

#ifdef GPU
    if (net.optimized_memory && params.train)
    {
        int k;
        for (k = 0; k < net.n; ++k) {
            layer l = net.layers[k];
            // delta GPU-memory optimization: net.optimized_memory == 1
            if (!l.keep_delta_gpu) {
                const size_t delta_size = l.outputs * l.batch; // l.steps
                if (net.max_delta_gpu_size < delta_size) {
                    net.max_delta_gpu_size = delta_size;
                    if (net.global_delta_gpu) cuda_free(net.global_delta_gpu);
                    if (net.state_delta_gpu) cuda_free(net.state_delta_gpu);
                    assert(net.max_delta_gpu_size > 0);
                    net.global_delta_gpu = (float*)cuda_make_array(NULL, net.max_delta_gpu_size);
                    net.state_delta_gpu = (float*)cuda_make_array(NULL, net.max_delta_gpu_size);
                }
                if (l.delta_gpu) {
                    if (net.optimized_memory >= 3) {}
                    else cuda_free(l.delta_gpu);
                }
                l.delta_gpu = net.global_delta_gpu;
            }

            // maximum optimization
            if (net.optimized_memory >= 3 && l.type != DROPOUT) {
                if (l.delta_gpu && l.keep_delta_gpu) {
                    //cuda_free(l.delta_gpu);   // already called above
                    l.delta_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch * l.outputs); // l.steps
                    //printf("\n\n PINNED DELTA GPU = %d \n", l.batch*l.outputs);
                }
            }
            net.layers[k] = l;
        }
    }
#endif
    set_train_only_bn(net); // set l.train_only_bn for all required layers
    net.outputs = get_network_output_size(net);
    net.output = get_network_output(net);
    avg_outputs = avg_outputs / avg_counter;
    fprintf(stdout, "Total BFLOPS %5.3f \n", bflops);
    fprintf(stdout, "avg_outputs = %d \n", avg_outputs);
#ifdef GPU
    get_cuda_stream();
    get_cuda_memcpy_stream();
    if (gpu_index >= 0)
    {
        int size = get_network_input_size(net) * net.batch;
        printf("network input size is : %d \n", size);
        net.input_state_gpu = cuda_make_array(0, size);
        if (cudaSuccess == cudaHostAlloc(&net.input_pinned_cpu, size * sizeof(float), cudaHostRegisterMapped)) net.input_pinned_cpu_flag = 1;
        else {
            cudaGetLastError(); // reset CUDA-error
            net.input_pinned_cpu = (float*)xcalloc(size, sizeof(float));
        }

        // pre-allocate memory for inference on Tensor Cores (fp16)
        *net.max_input16_size = 0;
        *net.max_output16_size = 0;
        if (net.cudnn_half) {
            *net.max_input16_size = max_inputs;
            CHECK_CUDA(cudaMalloc((void**)net.input16_gpu, *net.max_input16_size * sizeof(short))); //sizeof(half)
            *net.max_output16_size = max_outputs;
            CHECK_CUDA(cudaMalloc((void**)net.output16_gpu, *net.max_output16_size * sizeof(short))); //sizeof(half)
        }
        if (workspace_size) {
            fprintf(stderr, " Allocate additional workspace_size = %1.2f MB \n", (float)workspace_size / 1000000);
            net.workspace = cuda_make_array(0, workspace_size / sizeof(float) + 1);
        }
        else {
            net.workspace = (float*)xcalloc(1, workspace_size);
        }
    }
#else
    if (workspace_size) {
        net.workspace = (float*)xcalloc(1, workspace_size);
    }
#endif

    LAYER_TYPE lt = net.layers[net.n - 1].type;
    if ((net.w % 32 != 0 || net.h % 32 != 0) && (lt == YOLO || lt == REGION || lt == DETECTION)) {
        printf("\n Warning: width=%d and height=%d in cfg-file must be divisible by 32 for default networks Yolo v1/v2/v3!!! \n\n",
            net.w, net.h);
    }
    return net;

}

