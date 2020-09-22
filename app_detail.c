#include <stdlib.h>
#include "utils.h"
#include "dark_cuda.h"
#include "image_opencv.h"
#include "option_list.h"
#include "parser.h"
#include "data.h"

#ifdef GPUX
extern "C" {void run_detector(int argc, char** argv); }
#endif // GPU

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
        nets[k] = parse_network_cfg(cfg);

        //nets[k].benchmark_layers = benchmark_layers;
        if (weights) {
            load_weights(&nets[k], weights);
        }
        if (clear) {
            *nets[k].seen = 0;
            *nets[k].cur_iteration = 0;
        }
        nets[k].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];
    const int actual_batch_size = net.batch * net.subdivisions;
    if (actual_batch_size == 1) {
        printf("\n Error: You set incorrect value batch=1 for Training! You should set batch=64 subdivision=64 \n");
        getchar();
    }
    else if (actual_batch_size < 8) {
        printf("\n Warning: You set batch=%d lower than 64! It is recommended to set batch=64 subdivision=64 \n", actual_batch_size);
    }
    
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    data train, buffer;
    layer l = net.layers[net.n - 1];
    for (k = 0; k < net.n; k++) {
        layer lk = net.layers[k];
        if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION) {
            l = lk;
            printf(" Detection layer: %d - type = %d \n", k, l.type);
        }
    }
    int classes = l.classes;
    list* plist = get_paths(train_images);
    /*node* xxx = plist->front;
    while (xxx) {
        printf(" %s ", (char*)xxx->val);
        if (xxx->next)xxx = xxx->next;
        else break;
    }*/
    int imgs = net.batch * net.subdivisions * ngpus;
    int train_images_num = plist->size;
    char** paths = (char**)list_to_array(plist);
    const int init_w = net.w;
    const int init_h = net.h;
    const int init_b = net.batch;
    int iter_save, iter_save_last, iter_map;
    iter_save = get_current_iteration(net);
    iter_save_last = get_current_iteration(net);
    iter_map = get_current_iteration(net);
    float mean_average_precision = -1;
    float best_map = mean_average_precision;
    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.flip = net.flip;
    args.jitter = l.jitter;
    args.resize = l.resize;
    args.num_boxes = l.max_boxes;
    args.truth_size = l.truth_size;
    net.num_boxes = args.num_boxes;
    net.train_images_num = train_images_num;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    args.threads = 64;    // 16 or 64

    args.angle = net.angle;
    args.gaussian_noise = net.gaussian_noise;
    args.blur = net.blur;
    args.mixup = net.mixup;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    args.letter_box = net.letter_box;
    args.mosaic_bound = net.mosaic_bound;
    args.contrastive = net.contrastive;
    args.contrastive_jit_flip = net.contrastive_jit_flip;
    if (dont_show && show_imgs) show_imgs = 2;
    args.show_imgs = show_imgs;
#ifdef OPENCV
    //int num_threads = get_num_threads();
    //if(num_threads > 2) args.threads = get_num_threads() - 2;
    //args.threads = 6 * ngpus;   // 3 for - Amazon EC2 Tesla V100: p3.2xlarge (8 logical cores) - p3.16xlarge
    args.threads = 12 * ngpus;    // Ryzen 7 2700X (16 logical cores)
    mat_cv* img = NULL;
    float max_img_loss = net.max_chart_loss;
    int number_of_lines = 100;
    int img_size = 1000;
    char windows_name[100];
    sprintf(windows_name, "chart_%s.png", base);
    img = draw_train_chart(windows_name, max_img_loss, net.max_batches, number_of_lines, img_size, dont_show, chart_path);
#endif    //OPENCV 
    if (net.track) {
        args.track = net.track;
        args.augment_speed = net.augment_speed;
        if (net.sequential_subdivisions) args.threads = net.sequential_subdivisions * ngpus;
        else args.threads = net.subdivisions * ngpus;
        args.mini_batch = net.batch / net.time_steps;
        printf("\n Tracking! batch = %d, subdiv = %d, time_steps = %d, mini_batch = %d \n", net.batch, net.subdivisions, net.time_steps, args.mini_batch);
    }
    //printf(" imgs = %d \n", imgs);
    pthread_t  load_thread = load_data(args);
    int count = 0;
    double time_remaining, avg_time = -1, alpha_time = 0.01;
    while (get_current_iteration(net)<net.max_batches)
    {
        if (l.random && count++ % 10 == 0) {
            float rand_coef = 1.4;
            if (l.random != 1.0) rand_coef = l.random;
            printf("Resizing, random_coef = %.2f \n", rand_coef);
            float random_val = rand_scale(rand_coef);    // *x or /x
            int dim_w = roundl(random_val * init_w / net.resize_step + 1) * net.resize_step;
            int dim_h = roundl(random_val * init_h / net.resize_step + 1) * net.resize_step;
            if (random_val < 1 && (dim_w > init_w || dim_h > init_h)) dim_w = init_w, dim_h = init_h;

            int max_dim_w = roundl(rand_coef * init_w / net.resize_step + 1) * net.resize_step;
            int max_dim_h = roundl(rand_coef * init_h / net.resize_step + 1) * net.resize_step;

            // at the beginning (check if enough memory) and at the end (calc rolling mean/variance)
            if (avg_loss < 0 || get_current_iteration(net) > net.max_batches - 100) {
                dim_w = max_dim_w;
                dim_h = max_dim_h;
            }

            if (dim_w < net.resize_step) dim_w = net.resize_step;
            if (dim_h < net.resize_step) dim_h = net.resize_step;
            int dim_b = (init_b * max_dim_w * max_dim_h) / (dim_w * dim_h);
            int new_dim_b = (int)(dim_b * 0.8);
            if (new_dim_b > init_b) dim_b = new_dim_b;

            args.w = dim_w;
            args.h = dim_h;

            int k;
            if (net.dynamic_minibatch) {
                for (k = 0; k < ngpus; ++k) {
                    (*nets[k].seen) = init_b * net.subdivisions * get_current_iteration(net); // remove this line, when you will save to weights-file both: seen & cur_iteration
                    nets[k].batch = dim_b;
                    int j;
                    for (j = 0; j < nets[k].n; ++j)
                        nets[k].layers[j].batch = dim_b;
                }
                net.batch = dim_b;
                imgs = net.batch * net.subdivisions * ngpus;
                args.n = imgs;
                printf("\n %d x %d  (batch = %d) \n", dim_w, dim_h, net.batch);
            }
            else
                printf("\n %d x %d \n", dim_w, dim_h);

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for (k = 0; k < ngpus; ++k) {
                resize_network(nets + k, dim_w, dim_h);
            }
            net = nets[0];
        }
        double time = what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        if (net.track) {
            net.sequential_subdivisions = get_current_seq_subdivisions(net);
            args.threads = net.sequential_subdivisions * ngpus;
            printf(" sequential_subdivisions = %d, sequence = %d \n", net.sequential_subdivisions, get_sequence_value(net));
        }
        load_thread = load_data(args);

        const double load_time = (what_time_is_it_now() - time);
        printf("Loaded: %lf seconds", load_time);
        if (load_time > 0.1 && avg_loss > 0) printf(" - performance bottleneck on CPU or Disk HDD/SSD");
        printf("\n");
        time = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if (ngpus == 1) {
            int wait_key = (dont_show) ? 0 : 1;
            loss = train_network_waitkey(net, train, wait_key);
        }
        else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0 || avg_loss != avg_loss) avg_loss = loss;    // if(-inf or nan)
        avg_loss = avg_loss * .9 + loss * .1;
        const int iteration = get_current_iteration(net);
        printf("\n %d: %f, %f avg loss, %f rate, %lf seconds, %d images, %f hours left\n", iteration, loss, avg_loss, get_current_rate(net), (what_time_is_it_now() - time), iteration * imgs, avg_time);
        //8001: 0.595442, 0.595442 avg loss, 0.001300 rate, 13.209250 seconds, 512064 images, -1.000000 hours left
        fflush(stdout);
        break;
    }

#ifdef GPU
    if (ngpus > 1)sync_nets(nets,ngpus,0);
#endif // GPU
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    //save_weights(net, buff);
    printf("If you want to train from the beginning, then use flag in the end of training command: -clear \n");

#ifdef OPENCV
    release_mat(&img);
    destroy_all_windows_cv();
#endif

    // free memory
    pthread_join(load_thread, 0);
    free_data(buffer);

    free_load_threads(&args);

    free(base);
    free(paths);
    free_list_contents(plist);
    free_list(plist);

    free_list_contents_kvp(options);
    free_list(options);

    for (k = 0; k < ngpus; ++k) free_network(nets[k]);
    free(nets);
    //free_network(net);

    /*if (calc_map) {
        net_map.n = 0;
        free_network(net_map);
    }*/





    //getchar();
    //end of file
    if (gpus && gpu_list && ngpus > 1) free(gpus);
}
