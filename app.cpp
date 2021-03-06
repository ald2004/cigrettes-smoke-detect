#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <fstream>
#include <atomic>
#include <condition_variable>

#include <opencv2/opencv.hpp>
//#include <opencv2/ore/version.hpp>

#include "yolo_v2_class.hpp"


/*   add makefile
APPNAME=app
$(APPNAME): $(LIBNAMESO) include/yolo_v2_class.hpp src/app.cpp
        $(CPP) -std=c++11 $(COMMON) $(CFLAGS) -o $@ src/app.cpp $(LDFLAGS) -L ./ -l:$(LIBNAMESO)*/


std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> lines;
    if (!file.is_open()) return lines;
    for (std::string l; getline(file, l);) lines.push_back(l);
    std::cout << "object file " << filename << "names loaded!" << std::endl;
    return lines;
}

template <typename T>
class send_one_replaceable_object_t {
    T* a_ptr; //one frame , T* b_ptr,c_ptr,d_ptr
public:
    void send(T const& _obj) {
        std::unique_lock<std::mutex> l(mtx);
        condition.wait(l, [&]() {
            return a_ptr==NULL;
            });
        T* newobj = new T;
        *newobj = _obj;
        a_ptr = newobj;
        l.unlock(); condition.notify_all();
    }
    T receive() {
        std::unique_lock<std::mutex> l(mtx);
        std::unique_ptr<T> ptr;
            condition.wait(l, [&]() {
                return !(a_ptr == NULL);
                });
            ptr.reset(a_ptr);
            a_ptr = NULL;
        T obj = *ptr;
        l.unlock(); condition.notify_all();
        return obj;
    }
    send_one_replaceable_object_t(bool _sync) :a_ptr(NULL), condition() {}
private:
    std::condition_variable condition;
    mutable std::mutex mtx;
};

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
    int current_det_fps = -1, int current_cap_fps = -1)
{
    int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

    for (auto& i : result_vec) {
        cv::Scalar color = obj_id_to_color(i.obj_id);
        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
        if (obj_names.size() > i.obj_id) {
            std::string obj_name = obj_names[i.obj_id];
            if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
            cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
            max_width = std::max(max_width, (int)i.w + 2);
            //max_width = std::max(max_width, 283);
            std::string coords_3d;
            if (!std::isnan(i.z_3d)) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
                coords_3d = ss.str();
                cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
                int const max_width_3d = (text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
                if (max_width_3d > max_width) max_width = max_width_3d;
            }

            cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
                cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
                color, CV_FILLED, 8, 0);
            putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
            if (!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y - 1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
        }
    }
    if (current_det_fps >= 0 && current_cap_fps >= 0) {
        std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
        putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
    }
}
int main(int argc,char* argv[]) {
    std::string names_file = "cfg/smoke.names";
    std::string cfg_file = "cfg/smoke_small.cfg";
    std::string weights_file = "cfg/smoke_small_2000.weights";
    std::string filename="shenzhen.mp4";
    if (argc > 4) {
        names_file = argv[1];
        cfg_file = argv[2];
        weights_file = argv[3];
        filename = argv[4];

    }
    else if (argc > 1) filename = argv[1];
    if (!filename.size()) return 0;
    const float thres = (argc>5)?std::stof(argv[5]):.2;
    Detector detector(cfg_file, weights_file);
    cv::Size modelsize(detector.get_net_height(), detector.get_net_height());


    auto obj_names = objects_names_from_file(names_file);
    std::string outfile = "result.avi";
    bool const save_output_videofile = true;   // true - for history
    bool const send_network = false;        // true - for remote detection
    bool const use_kalman_filter = false;   // true - for stationary camera
    bool detection_sync = true;             // true - for video-file
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    if (ext == "avi" || ext == "mp4" || ext == "mjpg" || ext == "mov") {
        cv::Mat cur_frame;
        std::atomic<int> current_fps_cap(0), current_fps_det(0);
        std::atomic<int> fps_cap_counter(0), fps_det_counter(0);
        std::atomic<bool> exit_flag(false);
        std::chrono::steady_clock::time_point steady_start, steady_end;
        int  video_fps = 30;

        //if track : track_kalman_t track_kalman;
        try
        {
            cv::VideoCapture cap;
            cap.open(filename);
            if (!cap.isOpened()) {
                std::cout << "video not opened!" << std::endl;
            }
            else std::cout << "opened input video :" << filename << " \n";
            cap >> cur_frame;
            video_fps = cap.get(cv::CAP_PROP_FPS);
            //cv::Size const frame_size(960,960);
            cv::Size const frame_size = cur_frame.size();
            std::cout << "\n video size is" << frame_size << std::endl;
            cv::VideoWriter output_video;
            if (save_output_videofile) output_video.open(outfile, cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), std::max(35, video_fps), frame_size, true);
            if (output_video.isOpened()) {
                std::cout << "opened output video:" << outfile << std::endl;
                /*for (int i; i < 100; i++)
                    output_video << cur_frame;
                output_video.release();*/
            }

            struct detection_data_t {
                cv::Mat cap_frame; std::shared_ptr<image_t> det_image;
                std::vector<bbox_t> result_vec; cv::Mat draw_frame;
                bool new_detection; uint64_t frameid;
                bool exit_flag;
                detection_data_t() : new_detection(false), exit_flag(false) {}
            };
            const bool sync = detection_sync;
            //std::cout << std::endl << "sync.sync ....................." << sync << std::endl;
            send_one_replaceable_object_t<detection_data_t> cap2prepare(sync), cap2draw(sync),
                prepare2detect(sync), detect2draw(sync), draw2show(sync), draw2write(sync), draw2net(sync);
            std::thread t_cap, t_prepare, t_detect, t_post, t_draw, t_write, t_network;
            //capture new frame
            if (t_cap.joinable())t_cap.join();
            t_cap = std::thread([&]() {
                uint64_t frameid = 0;
                detection_data_t detection_data;

                do {
                    detection_data = detection_data_t();
                    cap >> detection_data.cap_frame;
                    fps_cap_counter++; detection_data.frameid = frameid++;
                    if (detection_data.cap_frame.empty() || exit_flag) {
                        std::cout << " exit_flag : detection_data.cap_frame.size" << detection_data.cap_frame.size() << std::endl;
                        detection_data.exit_flag = true;
                        detection_data.cap_frame = cv::Mat(frame_size, CV_8UC3);
                    }
                    if (!detection_sync) cap2draw.send(detection_data);
                    //std::cout << std::endl << "cap2draw.send ....................." << std::endl;
                    cap2prepare.send(detection_data);
                } while (!detection_data.exit_flag);
                std::cout << "t_cap exit!" << " \n";
                });
            //preprocess frame
            t_prepare = std::thread([&]() {
                std::shared_ptr<image_t> det_image;
                detection_data_t detection_data;

                do {
                    detection_data = cap2prepare.receive();
                    det_image = detector.mat_to_image_resize(detection_data.cap_frame);
                    detection_data.det_image = det_image;
                    //output_video << detection_data.cap_frame;
                    prepare2detect.send(detection_data);
                    //std::cout << std::endl << "prepare2detect.send ....................." << std::endl;
                } while (!detection_data.exit_flag);
                std::cout << "t_prepare exit! \n";
                });

            //det
            if (t_detect.joinable()) t_detect.join();
            t_detect = std::thread([&]() {
                std::shared_ptr<image_t> det_image;
                detection_data_t detection_data;
                do {
                    detection_data = prepare2detect.receive();
                    det_image = detection_data.det_image;
                    std::vector<bbox_t> result_vec;
                    if (det_image) {

                        result_vec = detector.detect_resized(*det_image, frame_size.width, frame_size.width, thres, true);
                        /*std::cout << "-+"<<" \n";
                        for (auto& i : result_vec)
                            std::cout << "x:" << i.x << "y:" << i.y << "w:" << i.w << "h:" << i.h;*/
                    }
                    fps_det_counter++;
                    detection_data.new_detection = true;
                    detection_data.result_vec = result_vec;
                    detect2draw.send(detection_data);
                    //std::cout << std::endl << "detect2draw.send ....................." << std::endl;
                } while (!detection_data.exit_flag);
                std::cout << "t_detect exit! \n";
                });

            //draw
            if (t_draw.joinable())t_draw.join();
            t_draw = std::thread([&]() {
                detection_data_t detection_data;

                do {
                    if (detection_sync)detection_data = detect2draw.receive();
                    cv::Mat cap_frame = detection_data.cap_frame;
                    cv::Mat draw_frame = detection_data.cap_frame.clone();
                    std::vector<bbox_t> result_vec = detection_data.result_vec;
                    int frame_story = std::max(5, current_fps_cap.load());
                    result_vec = detector.tracking_id(result_vec, true, frame_story, 40);
                    cv::Mat resizedmat;
                    cv::resize(draw_frame, resizedmat, modelsize);
                    draw_boxes(resizedmat, result_vec, obj_names, current_fps_det, current_fps_cap);
                    cv::resize(resizedmat, draw_frame, frame_size);
                    detection_data.result_vec = result_vec;
                    detection_data.draw_frame = draw_frame;
                    draw2show.send(detection_data);
                    //if (send_network)draw2net.send(detection_data);
                    //std::cout << std::endl << "draw2write.send xxxxxxxxxxxx....................." << std::endl;
                    if (output_video.isOpened()) {
                        //std::cout << std::endl << "draw2write.send ....................." << std::endl;
                        draw2write.send(detection_data);
                    }
                    //else std::cout << "output_vide is closed! \n";
                } while (!detection_data.exit_flag);
                std::cout << "t_draw exit! \n";
                });

            //wirte to file
            t_write = std::thread([&](){
                //std::cout << std::endl << "t_write start 11111111111111....................." << std::endl;
                if (output_video.isOpened()) {
                    detection_data_t detection_data;
                    cv::Mat output_frame;
                    //std::cout << std::endl << "t_write start ....................." << std::endl;
                    do {
                        detection_data = draw2write.receive();
                        if (detection_data.draw_frame.channels() == 4)cv::cvtColor(detection_data.draw_frame, output_frame, CV_RGBA2RGB);
                        else output_frame = detection_data.draw_frame;
                        //std::cout << std::endl << "t_write start 22222222222....................." << std::endl;
                        output_video << output_frame;
                    } while (!detection_data.exit_flag);
                    //output_video.release();
                }
                std::cout << " t_write exit! \n";
                });

            // send detection to the network
            /*t_network = std::thread([&]()
                {
                    if (send_network) {
                        detection_data_t detection_data;
                        do {
                            detection_data = draw2net.receive();

                            detector.send_json_http(detection_data.result_vec, obj_names, detection_data.frame_id, filename);

                        } while (!detection_data.exit_flag);
                    }
                    std::cout << " t_network exit \n";
                });*/

                // show detection
            detection_data_t detection_data;
            do {

                steady_end = std::chrono::steady_clock::now();
                float time_sec = std::chrono::duration<double>(steady_end - steady_start).count();
                if (time_sec >= 1) {
                    current_fps_det = fps_det_counter.load() / time_sec;
                    current_fps_cap = fps_cap_counter.load() / time_sec;
                    steady_start = steady_end;
                    fps_det_counter = 0;
                    fps_cap_counter = 0;
                }

                detection_data = draw2show.receive();
                //cv::Mat draw_frame = detection_data.draw_frame;
            } while (!detection_data.exit_flag);
            std::cout << " show detection exit \n";
            // wait for all threads
            if (t_cap.joinable()) t_cap.join();
            if (t_prepare.joinable()) t_prepare.join();
            if (t_detect.joinable()) t_detect.join();
            if (t_post.joinable()) t_post.join();
            if (t_draw.joinable()) t_draw.join();
            if (t_write.joinable()) t_write.join();
            if (t_network.joinable()) t_network.join();
            cap.release();
            filename.clear();
            try
            {
                output_video.release();
            }
            catch (const std::exception&)
            {

            }
        }
        catch (const std::exception& e) { std::cerr << "exception: " << e.what() << std::endl; }
        catch (...) { std::cerr << "unknown exception. \n"; }
    }
}
