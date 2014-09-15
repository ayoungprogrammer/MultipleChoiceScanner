#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

//g++ main.cpp -o main -I /usr/local/include/opencv -lopencv_core -lopencv_imgproc -lopencv_highgui

using namespace cv;
using namespace std;


cv::Point2f computeIntersect(cv::Vec4i a, cv::Vec4i b)
{
    int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
    int x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];

    if (float d = ((float)(x1-x2) * (y3-y4)) - ((y1-y2) * (x3-x4)))
    {
        cv::Point2f pt;
        pt.x = ((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) / d;
        pt.y = ((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) / d;
        return pt;
    }
    else
        return cv::Point2f(-1, -1);
}

bool comparator2(double a,double b){  
  return a<b;  
 }  
 bool comparator3(Vec3f a,Vec3f b){  
  return a[0]<b[0];  
 }  

bool comparator(Point2f a,Point2f b){  
  return a.x<b.x;  
  }  
 void sortCorners(std::vector<cv::Point2f>& corners, cv::Point2f center)  
 {  


   std::vector<cv::Point2f> top, bot;  
   for (int i = 0; i < corners.size(); i++)  
   {  
     if (corners[i].y < center.y)  
       top.push_back(corners[i]);  
     else  
       bot.push_back(corners[i]);  
   }  


  sort(top.begin(),top.end(),comparator);  
  sort(bot.begin(),bot.end(),comparator); 

   cv::Point2f tl = top[0];  
   cv::Point2f tr = top[top.size()-1];  
   cv::Point2f bl = bot[0];  
   cv::Point2f br = bot[bot.size()-1]; 
   corners.clear();  
   corners.push_back(tl);  
   corners.push_back(tr);  
   corners.push_back(br);  
   corners.push_back(bl);  
 }  


int main(int argc, char* argv[]){

  Mat img = imread("example.jpg",0);

  cv::Size size(3,3);  
  cv::GaussianBlur(img,img,size,0);  
  adaptiveThreshold(img, img,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,75,10);  
  cv::bitwise_not(img, img);    

  cv::Mat img2;  
  cvtColor(img,img2, CV_GRAY2RGB);  

  cv::Mat img3;  
  cvtColor(img,img3, CV_GRAY2RGB); 

  vector<Vec4i> lines;  
  HoughLinesP(img, lines, 1, CV_PI/180, 80, 400, 10);  
  for( size_t i = 0; i < lines.size(); i++ )  
  {  
   Vec4i l = lines[i];  
   line( img2, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);   
  }  

  imshow("example",img2);

  std::vector<cv::Point2f> corners;
  for (int i = 0; i < lines.size(); i++)
  {
      for (int j = i+1; j < lines.size(); j++)
      {
          cv::Point2f pt = computeIntersect(lines[i], lines[j]);
          if (pt.x >= 0 && pt.y >= 0 && pt.x < img.cols && pt.y < img.rows)
              corners.push_back(pt);
      }
  }

   // Get mass center  
  cv::Point2f center(0,0);  
  for (int i = 0; i < corners.size(); i++)  
  center += corners[i];  
  center *= (1. / corners.size()); 

  sortCorners(corners, center); 

  Rect r = boundingRect(corners); 
  cout<<r<<endl;
  cv::Mat quad = cv::Mat::zeros(r.height, r.width, CV_8UC3);  
  // Corners of the destination image  
  std::vector<cv::Point2f> quad_pts;  
  quad_pts.push_back(cv::Point2f(0, 0));  
  quad_pts.push_back(cv::Point2f(quad.cols, 0));  
  quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));  
  quad_pts.push_back(cv::Point2f(0, quad.rows));  

  // Get transformation matrix  
  cv::Mat transmtx = cv::getPerspectiveTransform(corners, quad_pts);  
  // Apply perspective transformation  
  cv::warpPerspective(img3, quad, transmtx, quad.size());  

  imshow("example2",quad);

  Mat cimg;
  
  cvtColor(quad,cimg, CV_BGR2GRAY);
  vector<Vec3f> circles;  
  HoughCircles(cimg, circles, CV_HOUGH_GRADIENT, 1, img.rows/8, 100, 75, 0, 0 );  
  for( size_t i = 0; i < circles.size(); i++ ){  
    Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    // circle center  
    circle( quad, center, 3, Scalar(0,255,0), -1, 8, 0 );
  }

  imshow("example4",quad);
  waitKey();

  double averR = 0;
  vector<double> row;
  vector<double> col;

  //Find rows and columns of circles for interpolation
  for(int i=0;i<circles.size();i++){  
    bool found = false;  
    int r = cvRound(circles[i][2]);
    averR += r;
    int x = cvRound(circles[i][0]);  
    int y = cvRound(circles[i][1]);
    for(int j=0;j<row.size();j++){
      double y2 = row[j];
      if(y - r < y2 && y + r > y2){
        found = true;
        break;
      }
    }
    if(!found){
      row.push_back(y);
    }
    found = false;
    for(int j=0;j<col.size();j++){
      double x2 = col[j];
      if(x - r < x2 && x + r > x2){
        found = true;
        break;
      }
    }
    if(!found){  
      col.push_back(x);
    }
  }

  averR /= circles.size();

  sort(row.begin(),row.end(),comparator2);
  sort(col.begin(),col.end(),comparator2);

  for(int i=0;i<row.size();i++){
    double max = 0;  
    double y = row[i];
    int ind = -1;
    for(int j=0;j<col.size();j++){
      double x = col[j];
      Point c(x,y);  

      //Use an actual circle if it exists
      for(int k=0;k<circles.size();k++){
        double x2 = circles[k][0];
        double y2 = circles[k][1];
        if(abs(y2-y)<averR && abs(x2-x)<averR){
          x = x2;
          y = y2;
        }
      }

      // circle outline  
      circle( quad, c, averR, Scalar(0,0,255), 3, 8, 0 );  
      Rect rect(x-averR,y-averR,2*averR,2*averR);  
      Mat submat = cimg(rect);  
      double p =(double)countNonZero(submat)/(submat.size().width*submat.size().height);  
      if(p>=0.3 && p>max){  
        max = p;  
        ind = j;  
      }  
    }
    if(ind==-1)printf("%d:-",i+1);  
    else printf("%d:%c",i+1,'A'+ind);  
    cout<<endl;  
  }  

  // circle outline*/  
  imshow("example3",quad);
  waitKey();
  return 0;
}