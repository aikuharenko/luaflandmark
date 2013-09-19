#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include "flandmark_detector.h"

extern "C" void* load_model(){

	FLANDMARK_Model* model = NULL;

	model = reinterpret_cast<void*>(flandmark_init("flandmark_model.dat"));
	if (model == NULL){
		printf("ERROR loading flandmark model!!!\n");
	}
	
	return model;

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


