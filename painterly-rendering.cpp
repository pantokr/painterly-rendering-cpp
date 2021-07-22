#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

using namespace std;
using namespace cv;

int STYLE;
int rCount = 7;

int THRESHOLD = 16; //임계값
int MIN_STROKE = 4; //최소 길이
int MAX_STROKE = 16; //최대 길이
double fc = 0.5; // 필터값

Mat paint(Mat, int[]); //바탕 함수
void paintLayer(Mat, Mat, int); //붓으로 칠하는 함수
double Difference(Scalar, Scalar); //이미지 차이 계산
double colorDistance(Scalar, Scalar); //색 차이 계산
vector<int*> styleCircle(int, int, int, Mat, Mat); //circle모드
vector<int*> styleStroke(int, int, int, Mat, Mat); //stroke모드
double gradientMag(Mat, int, int); //그래디언트 크기
double* gradientDirection(Mat, int, int); //그래디언트 방향

int main() {

	int R[] = { 48,32,16,8,4,3,2 }; //붓

	string str; //사진 위치 입력
	cout << "Source Location : ";
	cin >> str;
	Mat src = imread(str.c_str());
	Size size = src.size();
	Mat dst = Mat(size, CV_8UC3);
	int h = size.height;
	int w = size.width;

	int inp; //모드 입력

	cout << "0 : Circle, 1 : Stroke" << endl;
	cout << "Style : ";
	cin >> inp;

	if (inp == 0) {
		STYLE = 0;
	}
	else if (inp == 1) {
		STYLE = 1;
	}
	else {
		return 0;
	}

	imshow("src", src);

	dst = paint(src, R); //결과

	waitKey();

	str = "res_" + str;
	imwrite(str.c_str(), dst); //이미지 저장
	return 0;
}
Mat paint(Mat src, int R[]) {
	Size size = src.size(); //이미지 사이즈

	Mat canvas = Mat(size, CV_8UC3, Scalar(255, 255, 255));

	Mat referenceImage = Mat(size, CV_8UC3); //가우시안 블러용 이미지

	for (int i = 0; i < rCount; i++) {
		int gb = (R[i] / 2 * 2) + 1;
		GaussianBlur(src, referenceImage, Point(gb, gb), 1.5);// 블러 홀수화 적용
		paintLayer(canvas, referenceImage, R[i]); //본격 그리기 시작
		imshow("canvas", canvas);
		waitKey(1000); //1초마다 재생
	}

	return canvas;
}

void paintLayer(Mat canvas, Mat referenceImage, int R) {
	Size size = canvas.size();

	const int h = size.height;
	const int w = size.width;

	double** D; //각 점마다 색 차이를 구해 저장할 2차원 배열 변수
	D = new double* [h];
	for (int i = 0; i < h; i++) {
		D[i] = new double[w];
	}

	int grid = R; //그리드 붓 크기로
	if (grid < 1) {
		grid = 1;
	}

	// |(rgb1) - (rgb2)|
	for (int y = 0; y < h; y++) { //D 값에 이미지의 색 차이값을 계산해 저장
		for (int x = 0; x < w; x++) {
			Scalar difXY = referenceImage.at<Vec3b>(y, x);
			Scalar canXY = canvas.at<Vec3b>(y, x);

			D[y][x] = Difference(difXY, canXY);;
		}
	}

	vector<vector<int*>> S; //점 및 스트로크 저장 변수
	//Find the largest error point

	for (int y = 0; y < h; y += grid) {
		for (int x = 0; x < w; x += grid) {

			double M = 0; //평균 오차를 구하기 위한 변수 (논문 : areaError)

			double maxD = 0; //가장 오차가 큰 점의 차이값
			int Y = y, X = x; //가장 오차가 큰 점의 좌표

			for (int _y = y; _y < y + grid; _y++) {
				for (int _x = x; _x < x + grid; _x++) {
					if (_y >= h || _x >= w) {
						continue;
					}
					if (maxD < D[_y][_x]) {
						maxD = D[_y][_x]; // 오차 최대값과 그 좌표를 구하는 조건문
						Y = _y;
						X = _x;
					}
					M += D[_y][_x]; //영역의 모든 차이값을 더하고
				}
			}

			M /= (double)grid * grid; //영역의 크기로 나눠줌

			if (M > THRESHOLD) {
				if (STYLE == 0) {
					S.push_back(styleCircle(R, X, Y, referenceImage, canvas));//circle 모드일 때
				}
				else {
					S.push_back(styleStroke(R, X, Y, referenceImage, canvas));//stroke 모드일 때
				}
			}
		}
	}
	//점 순서 임의대로 섞고,
	random_device rd;
	shuffle(S.begin(), S.end(), default_random_engine(rd()));

	//모든 점 위치와 그 위치에 맞는 색으로 알맞은 작업
	if (STYLE == 0) { // 한 점씩만 찍고 종료
		for (int i = 0; i < S.size(); i++) {
			int Y = *(S[i][0] + 1);
			int X = *(S[i][0]);
			circle(canvas, Point(X, Y), R, referenceImage.at<Vec3b>(Y, X), FILLED);
		}
	}
	else {
		for (int i = 0; i < S.size(); i++) { //스트로크의 개수만큼 반복하는 반복문
			int Y = *(S[i][0] + 1); //각 스트로크 시작점 좌표 기억
			int X = *(S[i][0]);

			for (int j = 1; j < S[i].size(); j++) { //각 스트로크 언애 담긴 꼭짓점 개수만큼 반복하는 반복문

				int Y1 = *(S[i][j - 1] + 1); //이전점부터 현재점까지 스트로크
				int X1 = *(S[i][j - 1]);

				int Y2 = *(S[i][j] + 1);
				int X2 = *(S[i][j]);
				line(canvas, Point(X1, Y1), Point(X2, Y2), referenceImage.at<Vec3b>(Y, X), R);
			}
		}
	}

	//메모리 해제
	for (int i = 0; i < S.size(); i++) {
		for (int j = 0; j < S[i].size(); j++) {
			delete S[i][j];
		}
	}

	for (int i = 0; i < h; i++) {
		delete[] D[i];
	}
	delete[] D;
}

vector<int*> styleCircle(int R, int X, int Y, Mat referenceImage, Mat canvas) {
	vector<int*> s; //선택된 좌표값만 반환 형식에 맞게 변환하는 함수
	int* tmp = new int[2];
	tmp[0] = X; tmp[1] = Y;
	s.push_back(tmp);

	return s;
}

vector<int*> styleStroke(int R, int X, int Y, Mat referenceImage, Mat canvas) {
	Size size = referenceImage.size();

	int h = size.height;
	int w = size.width;

	Scalar strokeColor = referenceImage.at<Vec3b>(Y, X);

	vector<int*> K;
	int* _tmp = new int[2]; // 메모리 삭제를 막기 위한 동적할당
	_tmp[0] = X; _tmp[1] = Y; // 매개변수로 받아온 X, Y
	K.push_back(_tmp); //점 vector 배열에 삽입

	int tmp_x = X, tmp_y = Y; //논문에서의 (x, y), 선분의 꼭짓점을 나타냄
	double lastDx = 0, lastDy = 0; //초기의 이전 벡터값은 (0, 0)

	for (int i = 0; i < MAX_STROKE; i++) {
		int* tmp = new int[2];

		if (i > MIN_STROKE &&
			colorDistance(referenceImage.at<Vec3b>(tmp_y, tmp_x), referenceImage.at<Vec3b>(tmp_y, tmp_x))
			< colorDistance(referenceImage.at<Vec3b>(tmp_y, tmp_x), strokeColor)) {
			return K;
			//현재 블러 이미지에서의 점과 참조 캔버스에서의 점의 색 차이가 블러 이미지에서의 점과 초기 색과의 색 차보다 작아지면
			//즉, 새로 색을 갱신하는 것이 더 나아지면 stroke를 중단하고 선분 데이터 반환
		}


		if (gradientMag(referenceImage, tmp_x, tmp_y) == 0) { // 이 지점의 색에 그라데이션이 없고 단색이거나, 이미지 범위를 벗어나면 중단하고 선분 데이터 반환
			return K;
		}

		double* g = gradientDirection(referenceImage, tmp_x, tmp_y); //그라데이션의 방향 벡터

		double gx = *g;
		double gy = *(g + 1);

		double dx = (double)-gy; //수직 벡터
		double dy = (double)gx;

		delete g;

		if ((double)lastDx * dx + (double)lastDy * dy < 0) { //수직 벡터가 lastdx, lastdy와 이루는 각도가 수직 벡터 반대 방향 백터와 lastdx, lastdy와 이루는 각도 보다 작을 경우 
			dx = -dx; //수직 벡터를 반대 방향으로 전환
			dy = -dy;
		}

		dx = fc * dx + (1.0 - fc) * lastDx; //얼마나 부드럽게 표현할지 fc에 따라 방향이 미세 조정
		dy = fc * dy + (1.0 - fc) * lastDy;

		dx = dx / sqrt(dx * dx + dy * dy); //norm 구하는 과정
		dy = dy / sqrt(dx * dx + dy * dy);

		tmp_x = tmp_x + R * dx; //붓 반지름만큼 방향대로 이동한 후 그 점 저장
		tmp_y = tmp_y + R * dy;

		if (tmp_x < 0 || tmp_x >= w || tmp_y < 0 || tmp_y >= h) { //범위 이탈 시 종료
			return K;
		}
		lastDx = dx; lastDy = dy; //이전 단위 벡터 방향값을 가지고 있는 lastdx, lastdy

		tmp[0] = tmp_x; tmp[1] = tmp_y; //새로 이동한 점으로 vector<>에 넣어줌
		K.push_back(tmp);
	}

	return K;
}

double Difference(Scalar a, Scalar b) { //이미지 차이 이후 있을 색상 차이값과는 식이 다름
	double sum = 0;
	for (int k = 0; k < 3; k++) { //논문에 나온 3차원 두 점 간의 거리 구하는 식 인용 
		double distance = (a.val[k] - b.val[k]);
		distance *= distance;
		sum += distance;
	}
	return sqrt(sum);
}

double colorDistance(Scalar a, Scalar b) {
	double sum = 0;
	Scalar tmp;
	for (int k = 0; k < 3; k++) { // a에서 b를 뺀 Scalar 변수의 평균을 구해 
		tmp.val[k] = (a.val[k] - b.val[k]);
	}

	for (int k = 0; k < 3; k++) {
		sum += tmp.val[k];
	}

	sum /= 3;
	sum = sum > 0 ? sum : -sum; //절대값을 반환
	return sum;
}

double gradientMag(Mat referenceImage, int tmp_x, int tmp_y) {
	double* tmp = gradientDirection(referenceImage, tmp_x, tmp_y);

	double mag = (double)(*tmp) * (*tmp) + (double)(*(tmp + 1)) * (*(tmp + 1)); //피타고라스의 정리 x^2+y^2=z^2

	delete[] tmp;

	return sqrt(mag); //벡터 크기 반환
}

double* gradientDirection(Mat referenceImage, int tmp_x, int tmp_y) {
	Size size = referenceImage.size();

	int h = size.height;
	int w = size.width;

	double* tmp = new double[2];

	if (tmp_x + 1 >= w || tmp_x - 1 < 0 || tmp_y + 1 >= h || tmp_y - 1 < 0) //이미지 외곽이면 (0,0)벡터 반환 후 종료
	{
		tmp[0] = 0; tmp[1] = 0;
		return tmp;
	}

	double sumx1 = 0, sumx2 = 0; //주변 점들을 이용해 현재 점의 그라데이션 기울기 벡터를 구함
	double sumy1 = 0, sumy2 = 0;
	for (int k = 0; k < 3; k++) {
		sumx1 += referenceImage.at<Vec3b>(tmp_y, tmp_x + 1).val[k];
		sumx2 += referenceImage.at<Vec3b>(tmp_y, tmp_x - 1).val[k];
		sumy1 += referenceImage.at<Vec3b>(tmp_y + 1, tmp_x).val[k];
		sumy2 += referenceImage.at<Vec3b>(tmp_y - 1, tmp_x).val[k];
	}
	sumx1 /= 3; sumx2 /= 3; sumy1 /= 3; sumy2 /= 3; //각 점의 평균을 구해

	double gx = sumx1 - sumx2; //오른-왼 위-아래
	double gy = sumy1 - sumy2;

	tmp[0] = gx; tmp[1] = gy;
	return tmp; //방향벡터 반환
}