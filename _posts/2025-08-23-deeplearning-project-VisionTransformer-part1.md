---
layout: post
title: "Deep learning project - Vision Transformer part 1"
---

Some time has passed now since I started to follow AI, LLM, transformer, AE, VAE, etc. and watch the exploding trend of applications unfold within image processing, computer vision and computer graphics. It is fascinating what we accomplish once we develop new tools and start to play with them.

Since I have seen what the transformer architecture is capable of and its fundamental role in LLMs (for those who don’t know what those are they are the engine within ChatGPT/Gemini/Claude/Deepseek/etc.), I was thinking of jumping into the playground to get some hands-on experience with vision transformers. So there will be a series of this hobby project and this will serve as part 1. Note that this is not a tutorial, view it more as a diary of my work on a hobby project. This means that part 1 can have a lot of bugs which part 2 solves. Or if I find a cool solution to a problem I will write about it. 
I will try to post at least weekly or bi-weekly.

## Project goal

- Run a vision transformer model using ONNX and do something with a realtime video feed recorded by my camera e.g. segmentation or object detection.
- Build a simple GUI (e.g. with ImGUI) to do something interactively with the input or output of the transformer model. Render the result in realtime/interactive in a separate window.
- Use some C++17 and/or C++20 features for practice.
- Primarily focus on making it work on CPU. Then on GPU/CUDA.
- Later on when we finished we might consider combining multiple components e.g. let the vision transformer serve as a backbone and adding a front-end module or add some kind of embedding module, or other things to extend the project.

The goals are not set in stone and can be tweaked if we find something too cool to ignore that we want to focus on, in that case we will ditch the other stuff. We basically let whatever cool ideas come up guide the direction.

## Dependencies

I decided to use vcpkg - a C++ package manager. Last time I used it was like a decade ago. Starting with a simple description file vcpkg.json:

```json
{
  "name": "vit-app",
  "version-string": "0.0.1",
  "dependencies": [
    { "name": "opencv4", "features": ["gtk"] },
    { "name": "imgui", "features": ["glfw-binding", "opengl3-binding"] },
    { "name": "glfw3" },
    { "name": "glm" },
    { "name": "catch2" },
    { "name": "benchmark" },
    { "name": "spdlog" }
  ]
}
```

I use CMake to generate build files. This is how the CMakeLists.txt looks like currently:

```cmake
cmake_minimum_required(VERSION 3.22)
project(vit_app CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED PATHS /usr/lib/x86_64-linux-gnu/cmake/opencv4 /usr/share/opencv4 NO_DEFAULT_PATH)
find_package(glfw3 REQUIRED)
find_package(imgui CONFIG REQUIRED)

# Manual ONNX config 
if(DEFINED ENV{ONNXRUNTIME_DIR})
  set(ONNXRUNTIME_ROOT_PATH "$ENV{ONNXRUNTIME_DIR}")
  set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT_PATH}/include")
  set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT_PATH}/lib")
  
  find_library(ONNXRUNTIME_LIB 
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_LIB_DIR}
    NO_DEFAULT_PATH
  )
  
  if(ONNXRUNTIME_LIB)
    set(onnxruntime_FOUND TRUE)
  endif()
endif()

add_executable(vit_app src/main.cpp)
target_link_libraries(vit_app PRIVATE
  opencv_core opencv_imgproc opencv_highgui
  glfw imgui::imgui OpenGL::GL
)

if(onnxruntime_FOUND)
  target_compile_definitions(vit_app PRIVATE HAVE_ONNXRUNTIME=1)
  target_include_directories(vit_app PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
  target_link_libraries(vit_app PRIVATE ${ONNXRUNTIME_LIB})
endif()
```

As you might have noticed I had to set up a manual configuration of ONNX because find_package() didn't really find the correct libs since it kept looking in wrong directories.

I use Ninja to build the system. 

My simple main.cpp starts like this:

```cpp

#include <opencv2/opencv.hpp>
#include <iostream>

#ifdef HAVE_ONNXRUNTIME
    #include <onnxruntime_cxx_api.h>
#endif

int main() {

    std::cout << "ONNX: " << std::endl;

    auto providers = Ort::GetAvailableProviders();
    for (auto provider : providers) {
        std::cout << provider << std::endl;
    }

#ifdef HAVE_ONNXRUNTIME
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "hello");
    std::cout << "ONNX Runtime OK" << std::endl;
#else
    std::cout << "ONNX Runtime not linked (set ONNXRUNTIME_DIR to enable)" << std::endl;
#endif

    std::cout << "OpenCV: " << CV_VERSION << std::endl;
...
```

And I use OpenCV to connect to my camera like this:

```cpp
    cv::VideoCapture cap(0, cv::CAP_V4L2);

    if (!cap.isOpened()) { 

        std::cerr << "No camera found! Trying to create test image..." << std::endl;
        cv::Mat test(480, 640, CV_8UC3, cv::Scalar(0, 255, 0));
        cv::putText(test, "No camera - Press q to quit", {20,240},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, {255,255,255}, 2);
        cv::imshow("visiontransformer-app", test);

        while (cv::waitKey(0) != 'q');

        return 0;
    }
  
    // We force MJPG format i.e. JPEG for higher resolution of our camera
    // (otherwise it default to YUYV that is limited to 640x480 on this camera)
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    double actualWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double actualHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Camera opened successfully " << std::endl;
```

and reading the video stream like this:

```cpp
    while (true) {

        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Camera delivered empty frame" << std::endl;
            break;
        }

        std::cout << "Frame " << ++count << " size: " << frame.cols << "x" << frame.rows << std::endl;
        cv::putText(frame, "Press q to quit", {20,40},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);

        cv::imshow("visiontransformer-app", frame);

        int k = cv::waitKey(1);
        if (k == 'q' || k == 'Q') break;
        if (cv::getWindowProperty("visiontransformer-app", cv::WND_PROP_VISIBLE) < 1) break;
    }
...
```

We might play around and do something cool in raw format (UVY) later on. Althought it is tiny 640x480 since I use a cheap Logitech webcam for my digital meetings. But for now we will stick with compressed MJPG/JPEG streams which is capable to output full HD.

We can double-check the capacity using V4L2:

```bash
> v4l2-ctl --list-formats-ext -d /dev/video0

ioctl: VIDIOC_ENUM_FMT
	Type: Video Capture

	[0]: 'MJPG' (Motion-JPEG, compressed)
		Size: Discrete 1920x1080
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 1280x720
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 800x480
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 640x360
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 320x240
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 176x144
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 800x600
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 1920x1080
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.033s (30.000 fps)
	[1]: 'YUYV' (YUYV 4:2:2)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 640x360
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 320x240
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 176x144
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
			Interval: Discrete 0.033s (30.000 fps)
```

First phase done! We have OpenCV installed. ONNX installed and linked. Our camera rolling and video stream captured. A first setup looks like this:

<img width="510" height="310" alt="Screenshot from 2025-08-23" src="https://github.com/user-attachments/assets/7b3a0cfc-c342-4e8d-8f15-6f149fa8bd90" />

That was it for this time. Next time I will dig into ONNX a bit. Never worked with it before.
