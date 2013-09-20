#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include "flandmark_detector.h"

using namespace cv;
using namespace std;

IplImage* float_to_ipl(float *im, int width, int height){
	
	char* c_im = (char*)malloc(width * height * sizeof(char));

	for (int i = 0; i < width * height; i++)
		c_im[i] = (char)(255 * im[i]);

	IplImage* img_grayscale = cvCreateImageHeader(cvSize(width, height), IPL_DEPTH_8U, 1);
	img_grayscale->imageData = c_im;
	img_grayscale->imageDataOrigin = img_grayscale->imageData;
	
	return img_grayscale;
}

extern "C" void* load_model(){

	FLANDMARK_Model* model = NULL;

	model = reinterpret_cast<void*>(flandmark_init("flandmark_model.dat"));
	if (model == NULL){
		printf("ERROR loading flandmark model!!!\n");
	}
	
	return model;

}

void find_points(FLANDMARK_Model* model, IplImage* im, Rect bbox, float* res){
	
	//printf("finding points\n");	
	
	int bbox0[4];
	bbox0[0] = bbox.x;
	bbox0[1] = bbox.y;
	bbox0[2] = bbox.x + bbox.width - 1;
	bbox0[3] = bbox.y + bbox.height - 1;
	
	double* landmarks = (double*)malloc(2*model->data.options.M*sizeof(double));
	flandmark_detect(im, bbox0, model, landmarks);	

	for (int i = 0; i < 2 * model->data.options.M; i++){
		//printf("%f ", landmarks[i]);
		res[i] = (float) landmarks[i];
	}	
	//printf("\n");
	
	return landmarks;

}

extern "C" vector<Rect> find_faces(IplImage* im, int width, int height){
	
	//printf("fining faces\n");
		
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	CascadeClassifier face_cascade;
	
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); };	
	
	vector<Rect> faces;
	
	face_cascade.detectMultiScale( im, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	
	return faces;
}

extern "C" void find_faces_and_points(void* vmodel, float* f_im, int width, int height, float* res){
	
	//convert void* from torch to flandmark* model
	FLANDMARK_Model* model = (FLANDMARK_Model*) vmodel;
	
	//convert float* image to IplImage*
	IplImage* im = float_to_ipl(f_im, width, height);	
	
	//find faces in image
	vector<Rect> faces = find_faces(im, width, height);
	
	res[0] = faces.size();
	
	//find feature points on each face
	for( int i = 0; i < faces.size(); i++ )
	{
		 printf("%d %d %d %d\n", faces[i].x, faces[i].y, faces[i].width, faces[i].height);
		 res[i*20 + 1] = faces[i].x;
		 res[i*20 + 2]= faces[i].y;
		 res[i*20 + 3] = faces[i].width;
		 res[i*20 + 4] = faces[i].height;
		 
		 find_points(model, im, faces[i], &res[i * 20 + 5]);
	}
	
	//delete im data
	if (im->imageData != NULL){
		free(im->imageData);
		im->imageData = NULL;
	}
	cvReleaseImageHeader(&im);
	
}




extern "C" void detect(void* vmodel, float* f_im, int width, int height, float* res){

//	IplImage *img = cvLoadImage("/home/tema/Datasets/gml_VA_Train/photos/001/000003/photo_000003.jpg", 1);

	FLANDMARK_Model* model = (FLANDMARK_Model*) vmodel;
	
	char* c_im = (char*)malloc(width*height*sizeof(char));

	for (int i = 0; i < width * height; i++)
		c_im[i] = (char)(255 * f_im[i]);

	IplImage* img_grayscale = cvCreateImageHeader(cvSize(width, height), IPL_DEPTH_8U, 1);
	img_grayscale->imageData = c_im;
	img_grayscale->imageDataOrigin = img_grayscale->imageData;


//	IplImage *img_grayscale = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
//	cvCvtColor(img, img_grayscale, CV_BGR2GRAY);

	
	int bbox[] = {20, 20, 203, 203};

	double * landmarks = (double*)malloc(2*model->data.options.M*sizeof(double));
	flandmark_detect(img_grayscale, bbox, model, landmarks);	

	for (int i = 0; i < 2*model->data.options.M; i ++){
		res[i] = landmarks[i];
	}

/*	for (int i = 2; i < 2*model->data.options.M; i += 2)
    {
//        cvCircle(img_grayscale, cvPoint(int(landmarks[i]), int(landmarks[i+1])), 1, CV_SCALAR(255), CV_FILLED);
		printf("%f %f\n", landmarks[i], landmarks[i+1]);
    }
*/
	cvSaveImage("foo2.png", img_grayscale);

	if (c_im != NULL){
		free(c_im);
		c_im = NULL;
	}
		
	cvReleaseImageHeader( &img_grayscale );

	printf("hello\n");

}

extern "C" void free(void* vmodel){

	FLANDMARK_Model* model = (FLANDMARK_Model*) vmodel;
	flandmark_free(model);

}


