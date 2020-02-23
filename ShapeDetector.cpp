// ShapeDetector.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"



	


struct Shape
{
	std::string ShapeType = "";
	cv::Point TopLeft = cv::Point(0,0);
	cv::Point BottomRight = cv::Point(0, 0);
};

void addFielfToOutputFile(std::string &outputFile, std::string shapeType, int x_TL, int y_TL, int x_BR, int y_BR)
{
	std::string temp = "";

	temp = temp + "{\n";
	temp = temp + "\t\"boundingPoly\" : {\n";

	temp = temp + "\t\t\"vertices\" : [\n";

	temp = temp + "\t\t\t{\n";
	temp = temp + "\t\t\t\t\"x\" : " + std::to_string(x_TL) + ",\n";
	temp = temp + "\t\t\t\t\"y\" : " + std::to_string(y_TL) + "\n";
	temp = temp + "\t\t\t},\n";
	
	temp = temp + "\t\t\t{\n";
	temp = temp + "\t\t\t\t\"x\" : " + std::to_string(x_BR) + ",\n";
	temp = temp + "\t\t\t\t\"y\" : " + std::to_string(y_BR) + "\n";
	temp = temp + "\t\t\t}\n";
					

	temp = temp + "\t\t]\n";
	temp = temp + "\t},\n";
	temp = temp + "\t\"description\" : \"" + shapeType + "\"\n}\n";
	 
	outputFile = outputFile + temp;
}

cv::Rect findFigureBoundingBox(cv::Mat Image, cv::Vec3f Circle, std::vector<cv::Vec4i> lines)
{
	cv::Rect boundingBox;
	cv::Mat drawing = cv::Mat::zeros(Image.size(), CV_8UC1);

	if (Circle[2] > 0)
	{
		// draw circle
		cv::Point center = cv::Point(Circle[0], Circle[1]);
		// circle outline
		double radius = Circle[2];

		boundingBox = cv::Rect(center.x - radius, center.y - radius, 2 * radius, 2 * radius);
	}
	else if (lines.size() > 0)
	{
		// draw lines
		for (size_t i = 0; i < lines.size(); i++)
		{
			cv::Vec4i l = lines[i];
			line(drawing, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 255), 1, cv::LINE_4);
		}
		// calculate contour
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Point> contours_total;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(drawing, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

		// get BB
		if (contours.size() > 0)
		{
			for (int i = 0; i < contours.size(); i++)
			{
				for (int k = 0; k < contours[i].size(); k++)
				{
					contours_total.push_back(contours[i][k]);
				}
			}
			std::vector<cv::Point> contours_poly(contours_total.size());
			approxPolyDP(contours_total, contours_poly, 3, true);
			boundingBox = cv::boundingRect(contours_poly);
		}
	}

	
	return boundingBox;
}

cv::Mat calculateVariance(cv::Mat Image, bool bDebug)
{
	// local variance calculation
	cv::Mat Image_gray, dst;
	int kernel_size = 7;
	int scale = 10;
	int delta = 0;
	int ddepth = CV_16S;

	cv::cvtColor(Image, Image_gray, cv::COLOR_BGR2GRAY);

	/// Apply Laplace function
	cv::Mat abs_dst;
	cv::Laplacian(Image_gray, dst, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT);
	convertScaleAbs(dst, abs_dst);
	cv::Scalar mean, stddev; //0:1st channel, 1:2nd channel and 2:3rd channel
	meanStdDev(abs_dst, mean, stddev, cv::Mat());
	if (bDebug)
	{
		imshow("Laplacian", abs_dst);
		cv::waitKey(0);
	}
	return abs_dst;

}

cv::Mat detectAndFilterContours(cv::Mat Image_grey, cv::Mat Image, int contour_min_length, bool bDebug)
{
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(Image_grey, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	cv::Mat drawing = cv::Mat::zeros(Image_grey.size(), CV_8UC1);
	for (size_t i = 0; i < contours.size(); i++)
	{
		if(contours[i].size() > contour_min_length)
			drawContours(drawing, contours, (int)i, cv::Scalar(255, 255, 255), 2, cv::LINE_8, hierarchy, 0);
	}

	if (bDebug)
	{
		cv::imshow("Contours", drawing);
		cv::waitKey(0);
	}
	return drawing;
}

cv::Mat MorphOpen(cv::Mat Image_gray, int StructuringElementWidth, int StructuringElementHeight, bool bDebug)
{
	cv::Mat dst;

	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(StructuringElementWidth, StructuringElementHeight));
	cv::morphologyEx(Image_gray, dst, cv::MORPH_OPEN, element);
	if (bDebug)
	{
		cv::imshow("after morph op", dst);
		cv::waitKey(0);
	}
	return dst;
}

cv::Mat MorphClose(cv::Mat Image_gray, int StructuringElementWidth, int StructuringElementHeight, bool bDebug)
{
	cv::Mat dst;

	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(StructuringElementWidth, StructuringElementHeight));
	cv::morphologyEx(Image_gray, dst, cv::MORPH_CLOSE, element);
	if (bDebug)
	{
		cv::imshow("after morph op", dst);
		cv::waitKey(0);
	}
	return dst;
}

cv::Mat EdgeDetection(cv::Mat Image, bool bDebug)
{

	cv::Mat CannyEdgeMap;
	double Threshold_low = 100;
	double Threshold_high = 120;
	Canny(Image, CannyEdgeMap, Threshold_low, Threshold_high);

	if (bDebug)
	{
		cv::imshow("Edges current", CannyEdgeMap);
		cv::waitKey(0);
	}

	return CannyEdgeMap;
}

std::vector<cv::Vec3f> detectCircles(cv::Mat Image_gray, cv::Mat Image, bool bDebug)
{
	std::vector<cv::Vec3f> circles;

	double minDist = 60;
	// higher threshold of Canny Edge detector, lower threshold is twice smaller
	double p1UpperThreshold = 200;
	// the smaller it is, the more false circles may be detected
	double p2AccumulatorThreshold = 20;
	int minRadius = 30;
	int maxRadius = 0;
	// use gray image, not edge detected
	cv::HoughCircles(Image_gray, circles, cv::HOUGH_GRADIENT, 1, minDist, p1UpperThreshold, p2AccumulatorThreshold, minRadius, maxRadius);

	if (bDebug)
	{
		for (size_t i = 0; i < circles.size(); i++)
		{
			cv::Vec3i c = circles[i];
			cv::Point center = cv::Point(c[0], c[1]);
			// circle center
			circle(Image, center, 1, cv::Scalar(0, 100, 100), 1, cv::LINE_AA);
			// circle outline
			int radius = c[2];
			circle(Image, center, radius, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
		}
		cv::imshow("detected circles", Image);
		cv::waitKey(0);
	}
	return circles;
}

cv::Mat ColorThresholding(cv::Mat Image, int low_H, int high_H, bool bDebug)
{
	// color filtering
	const int max_value = 255;
	int low_S = 0;
	int low_V = 0;
	int high_S = max_value;
	int high_V = max_value;
	cv::Mat Image_HSV, Image_filtered;
	cv::cvtColor(Image, Image_HSV, cv::COLOR_BGR2HSV);
	// Detect the object based on HSV Range Values
	cv::inRange(Image_HSV, cv::Scalar(low_H, low_S, low_V), cv::Scalar(high_H, high_S, high_V), Image_filtered);
	if (bDebug)
	{
		cv::imshow("Filtered image", Image_filtered);
		cv::waitKey(0);
	}
	return Image_filtered;
}

cv::Mat ValueThresholding(cv::Mat Image, int low_V, int high_V, bool bDebug)
{
	// color filtering
	const int max_value = 255;
	int low_H = 0;
	int low_S = 0;
	int high_H = 180;
	int high_S = max_value;
	cv::Mat Image_HSV, Image_filtered;
	cv::cvtColor(Image, Image_HSV, cv::COLOR_BGR2HSV);
	// Detect the object based on HSV Range Values
	cv::inRange(Image_HSV, cv::Scalar(low_H, low_S, low_V), cv::Scalar(high_H, high_S, high_V), Image_filtered);
	if (bDebug)
	{
		cv::imshow("Filtered image", Image_filtered);
		cv::waitKey(0);
	}
	return Image_filtered;
}

std::vector<cv::Vec4i> detectLines(cv::Mat Image_gray, cv::Mat Image, bool bDebug)
{
	// LINES
				// Hough line detection

	std::vector<cv::Vec4i> ListHoughLines, ListHoughLinesFinal;
	//double MinLineLength = FSrcImage.size().height / 2.0;
	double MinLineLength = Image_gray.size().height / 10.0;
	double MaxLineGap = MinLineLength / 2.0;

	HoughLinesP(Image_gray, ListHoughLines, 1, CV_PI / 180, 50, MinLineLength, MaxLineGap);


	std::vector<cv::Vec4i> HorLines;
	std::vector<cv::Vec4i> VerLines;
	for (size_t i = 0; i < ListHoughLines.size(); i++)
	{
		cv::Vec4i l = ListHoughLines[i];
		if (abs(l[1] - l[3]) < 0.0002)
		{
			HorLines.push_back(l);
		}
		else if (abs(l[0] - l[2]) < 0.0002)
		{
			VerLines.push_back(l);
		}
	}

	// l[0] = x0
	// l[1] = y0
	// l[2] = x1
	// l[3] = y1

	// Find adjacent lines
	int th_dist = 3;
	for (size_t i = 0; i < HorLines.size(); i++)
	{
		for (size_t k = 0; k < VerLines.size(); k++)
		{
			if ((HorLines[i][0] - VerLines[k][0] < th_dist) && (HorLines[i][1] - VerLines[k][1] < th_dist))
			{
				//P0(line 1) == P0(line 2)
				ListHoughLinesFinal.push_back(HorLines[i]);
				ListHoughLinesFinal.push_back(VerLines[k]);
			}
			else if ((HorLines[i][0] - VerLines[k][2] < th_dist) && (HorLines[i][1] - VerLines[k][3] < th_dist))
			{
				//P0(line 1) == P1(line 2)
				ListHoughLinesFinal.push_back(HorLines[i]);
				ListHoughLinesFinal.push_back(VerLines[k]);
			}
			else if ((HorLines[i][2] - VerLines[k][0] < th_dist) && (HorLines[i][3] - VerLines[k][1] < th_dist))
			{
				//P1(line 1) == P0(line 2)
				ListHoughLinesFinal.push_back(HorLines[i]);
				ListHoughLinesFinal.push_back(VerLines[k]);
			}
			else if ((HorLines[i][2] - VerLines[k][2] < th_dist) && (HorLines[i][3] - VerLines[k][3] < th_dist))
			{
				//P1(line 1) == P1(line 2)
				ListHoughLinesFinal.push_back(HorLines[i]);
				ListHoughLinesFinal.push_back(VerLines[k]);
			}
		}

	}


	ListHoughLines = ListHoughLinesFinal;
	if (bDebug)
	{
		cv::Mat cdst = Image.clone();
		for (size_t i = 0; i < ListHoughLines.size(); i++)
		{
			cv::Vec4i l = ListHoughLines[i];
			line(cdst, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
		}
		cv::imshow("Lines", cdst);
		cv::waitKey(0);
	}
	return ListHoughLines;
}

int main()
{

	bool bDebug = false;
	cv::Mat Image = cv::imread("oriental_picture.png");
	if (bDebug)
	{
		cv::imshow("input", Image);
		cv::waitKey(1);
	}
	std::vector<Shape> ShapesDetected;

	// Blue circle detection
	cv::Mat Image_blue = ColorThresholding(Image, 120, 122, bDebug);
	Image_blue = MorphOpen(Image_blue, 10, 10, bDebug);
	std::vector<cv::Vec3f> blue_circles = detectCircles(Image_blue, Image, bDebug);
	std::vector<cv::Vec4i> lines_dummy;
	for (int i = 0; i < blue_circles.size(); i++)
	{
		Shape temp;
		temp.ShapeType = "Circle";
		cv::Rect tempRect = findFigureBoundingBox(Image, blue_circles[i], lines_dummy);
		temp.TopLeft = tempRect.tl();
		temp.BottomRight = tempRect.br();
		ShapesDetected.push_back(temp);
	}

	// Red square detection
	cv::Mat Image_red = ColorThresholding(Image, 0, 2, bDebug);
	Image_red = MorphOpen(Image_red, 10, 4, bDebug);
	Image_blue = MorphClose(Image_blue, 10, 4, bDebug);
	Image_red = EdgeDetection(Image_red, bDebug);
	std::vector<cv::Vec4i> red_lines = detectLines(Image_red, Image, bDebug);
	
	Shape temp;
	cv::Vec3f dummy_circle;
	temp.ShapeType = "Rectangle";
	cv::Rect tempRect = findFigureBoundingBox(Image, dummy_circle, red_lines);
	temp.TopLeft = tempRect.tl();
	temp.BottomRight = tempRect.br();
	ShapesDetected.push_back(temp);
	


	// White rectangle detection
	cv::Mat Image_gray;
	cv::cvtColor(Image, Image_gray, cv::COLOR_BGR2GRAY);
	cv::inRange(Image_gray, cv::Scalar(254), cv::Scalar(255), Image_gray);
	Image_gray = EdgeDetection(Image_gray, bDebug);
	Image_gray = detectAndFilterContours(Image_gray, Image, 50, bDebug);
	Image_gray = EdgeDetection(Image_gray, bDebug);
	std::vector<cv::Vec4i> all_lines = detectLines(Image_gray, Image, bDebug);
	
	temp.ShapeType = "Rectangle";
	tempRect = findFigureBoundingBox(Image, dummy_circle, all_lines);
	temp.TopLeft = tempRect.tl();
	temp.BottomRight = tempRect.br();
	ShapesDetected.push_back(temp);

	// Generate output
	bool showOutput = false;
	std::string outputFile = "";
	for (int i = 0; i < ShapesDetected.size(); i++)
	{
		addFielfToOutputFile(outputFile, ShapesDetected[i].ShapeType, ShapesDetected[i].TopLeft.x,
			ShapesDetected[i].TopLeft.y, ShapesDetected[i].BottomRight.x, ShapesDetected[i].BottomRight.y);
		if (showOutput)
		{
			cv::rectangle(Image, cv::Rect(ShapesDetected[i].TopLeft.x, ShapesDetected[i].TopLeft.y,
				abs(ShapesDetected[i].BottomRight.x - ShapesDetected[i].TopLeft.x),
				abs(ShapesDetected[i].BottomRight.y - ShapesDetected[i].TopLeft.y)),
				cv::Scalar(56, 255, 80), 3);
		}
	}
	if (showOutput)
	{
		cv::imshow("Detected Shapes", Image);
		cv::waitKey(0);
	}
	std::ofstream OutputGen("output.json");
	OutputGen << outputFile << std::endl;
	OutputGen.flush();
	OutputGen.close();
}
