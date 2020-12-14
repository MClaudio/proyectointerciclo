#include <iostream>
#include <cstdlib>
#include <cmath> 
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgcodecs/imgcodecs.hpp> 
#include <opencv2/video/video.hpp> 
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp> 

using namespace std;
using namespace cv;

int thresholdVal = 33, kVal = 13, dilatacionVal = 10, erocionVal = 1, aperturaVal = 1, cierreVal = 100, blackHatVal = 1, topHatVal = 100;

void eventoTrack(int v, void* pP) {
    
}

int main()
{
    VideoCapture video("C:\\Users\\claud\\Documents\\Projects\\C-C++\\Practicas\\videos\\video3.mp4");

    if (video.isOpened()) {
        Mat videoFrame;

        Mat videoDiffFrame;

        Mat ROIVideo;
        Mat gris;
        Mat frameAnterior;
        Mat frameActual;

        Mat frameMediana;
        Mat frameGauss;
        Mat bordeSobel;
        Mat bordeLaplacian;
 
        namedWindow("Video Original", WINDOW_AUTOSIZE);
        namedWindow("Video ROI", WINDOW_AUTOSIZE);
        namedWindow("Video ROI", WINDOW_AUTOSIZE);
        namedWindow("Video Mediana", WINDOW_AUTOSIZE);
        namedWindow("Video Gaussiano", WINDOW_AUTOSIZE); 
        namedWindow("Borde Sobel", WINDOW_AUTOSIZE); 
        namedWindow("Borde Laplacian", WINDOW_AUTOSIZE);

        createTrackbar("Threshold", "Video ROI", &thresholdVal, 255, eventoTrack, NULL);
        createTrackbar("K", "Video ROI", &kVal, 100, eventoTrack, NULL);
        createTrackbar("Dilatacion", "Video ROI", &dilatacionVal, 100, eventoTrack, NULL);
        createTrackbar("Erocion", "Video ROI", &erocionVal, 100, eventoTrack, NULL);
        createTrackbar("Apertura", "Video ROI", &aperturaVal, 100, eventoTrack, NULL);
        createTrackbar("Cierre", "Video ROI", &cierreVal, 100, eventoTrack, NULL);
        createTrackbar("Black Hat", "Video ROI", &blackHatVal, 100, eventoTrack, NULL);
        createTrackbar("Top Hat", "Video ROI", &topHatVal, 100, eventoTrack, NULL);

        while (true) {
            video >> videoFrame;
            resize(videoFrame, videoFrame, Size(), 0.50, 0.50);

            ROIVideo = videoFrame.clone();

            cvtColor(videoFrame, gris, COLOR_BGR2GRAY);
            frameActual = gris.clone();

            if (frameAnterior.cols == 0) {
                frameAnterior = gris.clone();
            }

            absdiff(frameAnterior, frameActual, videoDiffFrame);
            frameAnterior = frameActual.clone();

            threshold(videoDiffFrame, videoDiffFrame, thresholdVal, 255, THRESH_BINARY);

            if (erocionVal > 0) {
                Mat elementMORPH_ERODE = getStructuringElement(MORPH_CROSS, Size(erocionVal, erocionVal), Point(-1, -1));
                morphologyEx(videoDiffFrame, videoDiffFrame, MORPH_ERODE, elementMORPH_ERODE);

            }
            if (dilatacionVal > 0) {
                Mat elementMORPH_DILATE = getStructuringElement(MORPH_CROSS, Size(dilatacionVal, dilatacionVal), Point(-1, -1));

                morphologyEx(videoDiffFrame, videoDiffFrame, MORPH_DILATE, elementMORPH_DILATE);
            }

            if (blackHatVal > 0) {
                Mat elementMORPH_BLACKHAT = getStructuringElement(MORPH_CROSS, Size(blackHatVal, blackHatVal), Point(-1, -1));

                morphologyEx(videoDiffFrame, videoDiffFrame, MORPH_BLACKHAT, elementMORPH_BLACKHAT);
            }

            if (topHatVal > 0) {
                Mat elementMORPH_TOPHAT = getStructuringElement(MORPH_CROSS, Size(topHatVal, topHatVal), Point(-1, -1));

                morphologyEx(videoDiffFrame, videoDiffFrame, MORPH_TOPHAT, elementMORPH_TOPHAT);
            }

            if (aperturaVal > 0) {
                Mat elementMORPH_OPEN = getStructuringElement(MORPH_ELLIPSE, Size(aperturaVal, aperturaVal), Point(-1, -1));

                morphologyEx(videoDiffFrame, videoDiffFrame, MORPH_OPEN, elementMORPH_OPEN);
            }

            if (cierreVal > 0) {
                Mat elementMORPH_CLOSE = getStructuringElement(MORPH_RECT, Size(cierreVal, cierreVal), Point(-1, -1));

                morphologyEx(videoDiffFrame, videoDiffFrame, MORPH_CLOSE, elementMORPH_CLOSE);
            }

            if (kVal % 2 != 0) {
                medianBlur(videoDiffFrame, frameMediana, kVal);
                medianBlur(videoDiffFrame, videoDiffFrame, kVal);
            }

            for (int i = 1; i < kVal; i = i + 2)
            {
                GaussianBlur(videoDiffFrame, frameGauss, Size(i, i), 0, 0);
            }

            Sobel(videoDiffFrame, bordeSobel, CV_32F, 2, 0);
            Laplacian(videoDiffFrame, bordeLaplacian, CV_16S, 3);
            convertScaleAbs(bordeLaplacian, bordeLaplacian);

            int pixel = 0;
            for (int i = 0; i < videoFrame.rows; i++) {
                for (int j = 0; j < videoFrame.cols; j++) {
                    pixel = (int)videoDiffFrame.at<uchar>(i, j);
                    if (pixel != 0) {
                        //cout << "Pixel " << pixel << endl; 
                        break;
                    }
                    ROIVideo.at<Vec3b>(i, j) = videoDiffFrame.at<Vec3b>();
                        
                }
                for (int j = videoDiffFrame.cols - 1; j >= 0; j--) {
                    int pixel = (int)videoDiffFrame.at<uchar>(i, j);
                    if (pixel != 0) {
                        break;
                    }
                    ROIVideo.at<Vec3b>(i, j) = videoDiffFrame.at<Vec3b>();
  
                }
            }
   
            if (videoFrame.rows == 0 || videoFrame.cols == 0)
                break;

            imshow("Video Original", videoFrame);
            imshow("Video Diff", videoDiffFrame);
            imshow("Video ROI", ROIVideo);
            imshow("Video Mediana", frameMediana);
            imshow("Video Gaussiano", frameGauss);
            imshow("Borde Sobel", bordeSobel);
            imshow("Borde Laplacian", bordeLaplacian);
            if (waitKey(23) == 27)
                break;
        }
        destroyAllWindows();
    }

 
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
