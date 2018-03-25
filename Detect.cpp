#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

class detector {

    public:
        detector();
        detector(Mat src_frame)
            {
                src_frame.copyTo(frame2);
            }
        
        void detect();
        void detectFaceEyes();
        void detectUpperBody();
        void detectLowerBody();
        void detectFullBody();
        void detectSmile();
        void rotateImg();
        int checkCascade();
        

    private:

        int deg;

        Mat frame;
        Mat frame2;
        Mat frame_gray;

        int out[24][4];

        std::vector<Rect> faces;
        std::vector<Rect> eyes;
        std::vector<Rect> profile_eyes;
        std::vector<Rect> upperbody;
        std::vector<Rect> lowerbody;
        std::vector<Rect> fullbody;
        std::vector<Rect> smile;
        std::vector<Rect> profile;
        
        const String face_cascade_name = "haarcascade_frontalface_alt2.xml";
        const String profileface_cascade_name = "haarcascade_profileface.xml";
        const String eyes_cascade_name = "haarcascade_eye.xml";
        const String upper_cascade_name = "HS.xml";
        const String lower_cascade_name = "haarcascade_lowerbody.xml";
        const String full_cascade_name = "haarcascade_fullbody.xml";
        const String smile_cascade_name = "Mouth.xml";

        CascadeClassifier face_cascade;
        CascadeClassifier profileface_cascade;
        CascadeClassifier eyes_cascade;
        CascadeClassifier upper_cascade;
        CascadeClassifier full_cascade;
        CascadeClassifier lower_cascade;
        CascadeClassifier smile_cascade;

        enum checks { 
            FACE,
            UPPERBODY, 
            LOWERBODY, 
            FULLBODY,
            SMILE 
            } current_check;

};

int detector::checkCascade()
{
    if( !face_cascade.load(face_cascade_name) ) {
        printf("--(!)Error loading face cascade\n");
        return -1;
    }
    if( !eyes_cascade.load(eyes_cascade_name) ) {
        printf("--(!)Error loading eyes cascade\n");
        return -1;
    }
    if( !upper_cascade.load(upper_cascade_name) ) {
        printf("--(!)Error loading upper body cascade\n");
        return -1;
    }
    if( !full_cascade.load(full_cascade_name) ) {
        printf("--(!)Error loading full body cascade\n");
        return -1;
    }
    if( !lower_cascade.load(lower_cascade_name) ) {
        printf("--(!)Error loading lower body cascade\n");
        return -1;
    }
    if( !full_cascade.load(full_cascade_name) ) {
        printf("--(!)Error loading eyes cascade\n");
        return -1;
    }
    if( !smile_cascade.load(smile_cascade_name) ) {
        printf("--(!)Error loading mouth cascade\n");
        return -1;
    }
    if( !profileface_cascade.load(profileface_cascade_name) ) {
        printf("--(!)Error loading profileface cascade\n");
        return -1;
    }
}

void detector::detectFaceEyes()
{
    current_check = FACE;

    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(24, 24) );
    profileface_cascade.detectMultiScale( frame_gray, profile, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(24, 24) );
    

    for( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        Mat faceROI = frame_gray( faces[i] );
        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(20, 20) );
        
        for( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 255, 0 ), 4, 8, 0 );
        }
    }

    for( size_t i = 0; i < profile.size(); i++ )
    {
        Point center( profile[i].x + profile[i].width/2, profile[i].y + profile[i].height/2 );
        ellipse( frame, center, Size( profile[i].width/2, profile[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        Mat faceROI = frame_gray( profile[i] );
        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, profile_eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(20, 20) );

        for( size_t j = 0; j < profile_eyes.size(); j++ )
        {
            Point eye_center( profile[i].x + profile_eyes[j].x + profile_eyes[j].width/2, profile[i].y + profile_eyes[j].y + profile_eyes[j].height/2 );
            int radius = cvRound( (profile_eyes[j].width + profile_eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 255, 0 ), 4, 8, 0 );
        }
    }
    if ((faces.size() >= 1 && eyes.size() >=1) || (profile.size() >= 1 && profile_eyes.size()) >=1)
        out[deg][FACE] = 2;
    else if (faces.size() >= 1 || profile.size() >= 1)
        out[deg][FACE] = 1;
    else
        out[deg][FACE] = 0;
    
}

void detector::detectUpperBody()
{   
    current_check = UPPERBODY;
    
    upper_cascade.detectMultiScale( frame_gray, upperbody, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(22, 18) );
    
    for( size_t i = 0; i < upperbody.size(); i++ )
    {
        Point center( upperbody[i].x + upperbody[i].width/2, upperbody[i].y + upperbody[i].height/2 );
        ellipse( frame, center, Size( upperbody[i].width/2, upperbody[i].height/2), 0, 0, 360, Scalar( 255, 255, 255 ), 4, 8, 0 );
    }
    
    if (upperbody.size() >= 1)
        out[deg][UPPERBODY] = 1;
    else
        out[deg][UPPERBODY] = 0;
}

void detector::detectLowerBody()
{
    current_check = LOWERBODY;

    lower_cascade.detectMultiScale( frame_gray, lowerbody, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(23, 19) );
    
    for( size_t i = 0; i < lowerbody.size(); i++ )
        {
            Point center( lowerbody[i].x + lowerbody[i].width/2, lowerbody[i].y + lowerbody[i].height/2 );
            ellipse( frame, center, Size( lowerbody[i].width/2, lowerbody[i].height/2), 0, 0, 360, Scalar( 0, 0, 255 ), 4, 8, 0 );
        }
    
    if (lowerbody.size() >= 1)
        out[deg][LOWERBODY] = 1;
    else
        out[deg][LOWERBODY] = 0;
    
}

void detector::detectFullBody()
{
    current_check = FULLBODY;

    full_cascade.detectMultiScale( frame_gray, fullbody, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(28, 14) );
    
    for( size_t i = 0; i < fullbody.size(); i++ )
        {
            Point center( fullbody[i].x + fullbody[i].width/2, fullbody[i].y + fullbody[i].height/2 );
            ellipse( frame, center, Size( fullbody[i].width/2, fullbody[i].height/2), 0, 0, 360, Scalar( 0, 0, 255 ), 4, 8, 0 );
        }
    
    if (fullbody.size() >= 1)
        out[deg][FULLBODY] = 1;
    else
        out[deg][FULLBODY] = 0;
    
}

void detector::detectSmile()
{
    current_check = SMILE;

    smile_cascade.detectMultiScale( frame_gray, smile, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    
    for( size_t i = 0; i < smile.size(); i++ )
    {
        Point center( smile[i].x + smile[i].width/2, smile[i].y + smile[i].height/2 );
        ellipse( frame, center, Size( smile[i].width/2, smile[i].height/2), 0, 0, 360, Scalar( 255, 255, 255 ), 4, 8, 0 );
    }
    
    if (smile.size() >= 1)
        out[deg][SMILE] = 1;
    else
        out[deg][SMILE] = 0;
}

void detector::rotateImg() {

    Point2f pt(frame2.cols/2., frame2.rows/2.);    
    Mat r = getRotationMatrix2D(pt, (double)15*deg, 1.0);
    warpAffine(frame2, frame, r, Size(frame2.cols, frame2.rows));
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

}

void detector::detect()
{
    int sum = 0;
    int flag = 0;
    for (deg = 0 ; deg < 1 ; deg++) 
    {
        rotateImg();
        detectFaceEyes();
        detectUpperBody();
        detectLowerBody();
        detectFullBody();
        detectSmile();
    }
    imshow("OUT",frame);
    for (int j = 0; j < 5 ; j++)
    {
        sum += out[0][j];
        printf("%d\n",out[0][j]);
    }
    imshow("OUT",frame);
    waitKey(0);
    if (sum >= 3)
        printf("Human Found\n\n");
    else
        printf("Human Not Found.\n\n");
}

int main(int argc, char* argv[])
{
    Mat src_frame;
    if (argc > 2)
        printf("Wrong no. of arguments. Please enter file location as an argument\n");
    else if (argc == 1)
        cin.getline(argv[1],255,'\n');
    else 
        src_frame = imread(argv[1], 1);
    
    detector detect_body(src_frame);
    
    if (detect_body.checkCascade() == -1) {
        printf("Exiting...\n");
        exit(-1);
    }

    detect_body.detect();
    return 0;
}