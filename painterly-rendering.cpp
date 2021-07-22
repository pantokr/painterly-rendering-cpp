#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

using namespace std;
using namespace cv;

int STYLE;
int rCount = 7;

int THRESHOLD = 16; //�Ӱ谪
int MIN_STROKE = 4; //�ּ� ����
int MAX_STROKE = 16; //�ִ� ����
double fc = 0.5; // ���Ͱ�

Mat paint(Mat, int[]); //���� �Լ�
void paintLayer(Mat, Mat, int); //������ ĥ�ϴ� �Լ�
double Difference(Scalar, Scalar); //�̹��� ���� ���
double colorDistance(Scalar, Scalar); //�� ���� ���
vector<int*> styleCircle(int, int, int, Mat, Mat); //circle���
vector<int*> styleStroke(int, int, int, Mat, Mat); //stroke���
double gradientMag(Mat, int, int); //�׷����Ʈ ũ��
double* gradientDirection(Mat, int, int); //�׷����Ʈ ����

int main() {

	int R[] = { 48,32,16,8,4,3,2 }; //��

	string str; //���� ��ġ �Է�
	cout << "Source Location : ";
	cin >> str;
	Mat src = imread(str.c_str());
	Size size = src.size();
	Mat dst = Mat(size, CV_8UC3);
	int h = size.height;
	int w = size.width;

	int inp; //��� �Է�

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

	dst = paint(src, R); //���

	waitKey();

	str = "res_" + str;
	imwrite(str.c_str(), dst); //�̹��� ����
	return 0;
}
Mat paint(Mat src, int R[]) {
	Size size = src.size(); //�̹��� ������

	Mat canvas = Mat(size, CV_8UC3, Scalar(255, 255, 255));

	Mat referenceImage = Mat(size, CV_8UC3); //����þ� ���� �̹���

	for (int i = 0; i < rCount; i++) {
		int gb = (R[i] / 2 * 2) + 1;
		GaussianBlur(src, referenceImage, Point(gb, gb), 1.5);// �� Ȧ��ȭ ����
		paintLayer(canvas, referenceImage, R[i]); //���� �׸��� ����
		imshow("canvas", canvas);
		waitKey(1000); //1�ʸ��� ���
	}

	return canvas;
}

void paintLayer(Mat canvas, Mat referenceImage, int R) {
	Size size = canvas.size();

	const int h = size.height;
	const int w = size.width;

	double** D; //�� ������ �� ���̸� ���� ������ 2���� �迭 ����
	D = new double* [h];
	for (int i = 0; i < h; i++) {
		D[i] = new double[w];
	}

	int grid = R; //�׸��� �� ũ���
	if (grid < 1) {
		grid = 1;
	}

	// |(rgb1) - (rgb2)|
	for (int y = 0; y < h; y++) { //D ���� �̹����� �� ���̰��� ����� ����
		for (int x = 0; x < w; x++) {
			Scalar difXY = referenceImage.at<Vec3b>(y, x);
			Scalar canXY = canvas.at<Vec3b>(y, x);

			D[y][x] = Difference(difXY, canXY);;
		}
	}

	vector<vector<int*>> S; //�� �� ��Ʈ��ũ ���� ����
	//Find the largest error point

	for (int y = 0; y < h; y += grid) {
		for (int x = 0; x < w; x += grid) {

			double M = 0; //��� ������ ���ϱ� ���� ���� (�� : areaError)

			double maxD = 0; //���� ������ ū ���� ���̰�
			int Y = y, X = x; //���� ������ ū ���� ��ǥ

			for (int _y = y; _y < y + grid; _y++) {
				for (int _x = x; _x < x + grid; _x++) {
					if (_y >= h || _x >= w) {
						continue;
					}
					if (maxD < D[_y][_x]) {
						maxD = D[_y][_x]; // ���� �ִ밪�� �� ��ǥ�� ���ϴ� ���ǹ�
						Y = _y;
						X = _x;
					}
					M += D[_y][_x]; //������ ��� ���̰��� ���ϰ�
				}
			}

			M /= (double)grid * grid; //������ ũ��� ������

			if (M > THRESHOLD) {
				if (STYLE == 0) {
					S.push_back(styleCircle(R, X, Y, referenceImage, canvas));//circle ����� ��
				}
				else {
					S.push_back(styleStroke(R, X, Y, referenceImage, canvas));//stroke ����� ��
				}
			}
		}
	}
	//�� ���� ���Ǵ�� ����,
	random_device rd;
	shuffle(S.begin(), S.end(), default_random_engine(rd()));

	//��� �� ��ġ�� �� ��ġ�� �´� ������ �˸��� �۾�
	if (STYLE == 0) { // �� ������ ��� ����
		for (int i = 0; i < S.size(); i++) {
			int Y = *(S[i][0] + 1);
			int X = *(S[i][0]);
			circle(canvas, Point(X, Y), R, referenceImage.at<Vec3b>(Y, X), FILLED);
		}
	}
	else {
		for (int i = 0; i < S.size(); i++) { //��Ʈ��ũ�� ������ŭ �ݺ��ϴ� �ݺ���
			int Y = *(S[i][0] + 1); //�� ��Ʈ��ũ ������ ��ǥ ���
			int X = *(S[i][0]);

			for (int j = 1; j < S[i].size(); j++) { //�� ��Ʈ��ũ ��� ��� ������ ������ŭ �ݺ��ϴ� �ݺ���

				int Y1 = *(S[i][j - 1] + 1); //���������� ���������� ��Ʈ��ũ
				int X1 = *(S[i][j - 1]);

				int Y2 = *(S[i][j] + 1);
				int X2 = *(S[i][j]);
				line(canvas, Point(X1, Y1), Point(X2, Y2), referenceImage.at<Vec3b>(Y, X), R);
			}
		}
	}

	//�޸� ����
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
	vector<int*> s; //���õ� ��ǥ���� ��ȯ ���Ŀ� �°� ��ȯ�ϴ� �Լ�
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
	int* _tmp = new int[2]; // �޸� ������ ���� ���� �����Ҵ�
	_tmp[0] = X; _tmp[1] = Y; // �Ű������� �޾ƿ� X, Y
	K.push_back(_tmp); //�� vector �迭�� ����

	int tmp_x = X, tmp_y = Y; //�������� (x, y), ������ �������� ��Ÿ��
	double lastDx = 0, lastDy = 0; //�ʱ��� ���� ���Ͱ��� (0, 0)

	for (int i = 0; i < MAX_STROKE; i++) {
		int* tmp = new int[2];

		if (i > MIN_STROKE &&
			colorDistance(referenceImage.at<Vec3b>(tmp_y, tmp_x), referenceImage.at<Vec3b>(tmp_y, tmp_x))
			< colorDistance(referenceImage.at<Vec3b>(tmp_y, tmp_x), strokeColor)) {
			return K;
			//���� �� �̹��������� ���� ���� ĵ���������� ���� �� ���̰� �� �̹��������� ���� �ʱ� ������ �� ������ �۾�����
			//��, ���� ���� �����ϴ� ���� �� �������� stroke�� �ߴ��ϰ� ���� ������ ��ȯ
		}


		if (gradientMag(referenceImage, tmp_x, tmp_y) == 0) { // �� ������ ���� �׶��̼��� ���� �ܻ��̰ų�, �̹��� ������ ����� �ߴ��ϰ� ���� ������ ��ȯ
			return K;
		}

		double* g = gradientDirection(referenceImage, tmp_x, tmp_y); //�׶��̼��� ���� ����

		double gx = *g;
		double gy = *(g + 1);

		double dx = (double)-gy; //���� ����
		double dy = (double)gx;

		delete g;

		if ((double)lastDx * dx + (double)lastDy * dy < 0) { //���� ���Ͱ� lastdx, lastdy�� �̷�� ������ ���� ���� �ݴ� ���� ���Ϳ� lastdx, lastdy�� �̷�� ���� ���� ���� ��� 
			dx = -dx; //���� ���͸� �ݴ� �������� ��ȯ
			dy = -dy;
		}

		dx = fc * dx + (1.0 - fc) * lastDx; //�󸶳� �ε巴�� ǥ������ fc�� ���� ������ �̼� ����
		dy = fc * dy + (1.0 - fc) * lastDy;

		dx = dx / sqrt(dx * dx + dy * dy); //norm ���ϴ� ����
		dy = dy / sqrt(dx * dx + dy * dy);

		tmp_x = tmp_x + R * dx; //�� ��������ŭ ������ �̵��� �� �� �� ����
		tmp_y = tmp_y + R * dy;

		if (tmp_x < 0 || tmp_x >= w || tmp_y < 0 || tmp_y >= h) { //���� ��Ż �� ����
			return K;
		}
		lastDx = dx; lastDy = dy; //���� ���� ���� ���Ⱚ�� ������ �ִ� lastdx, lastdy

		tmp[0] = tmp_x; tmp[1] = tmp_y; //���� �̵��� ������ vector<>�� �־���
		K.push_back(tmp);
	}

	return K;
}

double Difference(Scalar a, Scalar b) { //�̹��� ���� ���� ���� ���� ���̰����� ���� �ٸ�
	double sum = 0;
	for (int k = 0; k < 3; k++) { //���� ���� 3���� �� �� ���� �Ÿ� ���ϴ� �� �ο� 
		double distance = (a.val[k] - b.val[k]);
		distance *= distance;
		sum += distance;
	}
	return sqrt(sum);
}

double colorDistance(Scalar a, Scalar b) {
	double sum = 0;
	Scalar tmp;
	for (int k = 0; k < 3; k++) { // a���� b�� �� Scalar ������ ����� ���� 
		tmp.val[k] = (a.val[k] - b.val[k]);
	}

	for (int k = 0; k < 3; k++) {
		sum += tmp.val[k];
	}

	sum /= 3;
	sum = sum > 0 ? sum : -sum; //���밪�� ��ȯ
	return sum;
}

double gradientMag(Mat referenceImage, int tmp_x, int tmp_y) {
	double* tmp = gradientDirection(referenceImage, tmp_x, tmp_y);

	double mag = (double)(*tmp) * (*tmp) + (double)(*(tmp + 1)) * (*(tmp + 1)); //��Ÿ����� ���� x^2+y^2=z^2

	delete[] tmp;

	return sqrt(mag); //���� ũ�� ��ȯ
}

double* gradientDirection(Mat referenceImage, int tmp_x, int tmp_y) {
	Size size = referenceImage.size();

	int h = size.height;
	int w = size.width;

	double* tmp = new double[2];

	if (tmp_x + 1 >= w || tmp_x - 1 < 0 || tmp_y + 1 >= h || tmp_y - 1 < 0) //�̹��� �ܰ��̸� (0,0)���� ��ȯ �� ����
	{
		tmp[0] = 0; tmp[1] = 0;
		return tmp;
	}

	double sumx1 = 0, sumx2 = 0; //�ֺ� ������ �̿��� ���� ���� �׶��̼� ���� ���͸� ����
	double sumy1 = 0, sumy2 = 0;
	for (int k = 0; k < 3; k++) {
		sumx1 += referenceImage.at<Vec3b>(tmp_y, tmp_x + 1).val[k];
		sumx2 += referenceImage.at<Vec3b>(tmp_y, tmp_x - 1).val[k];
		sumy1 += referenceImage.at<Vec3b>(tmp_y + 1, tmp_x).val[k];
		sumy2 += referenceImage.at<Vec3b>(tmp_y - 1, tmp_x).val[k];
	}
	sumx1 /= 3; sumx2 /= 3; sumy1 /= 3; sumy2 /= 3; //�� ���� ����� ����

	double gx = sumx1 - sumx2; //����-�� ��-�Ʒ�
	double gy = sumy1 - sumy2;

	tmp[0] = gx; tmp[1] = gy;
	return tmp; //���⺤�� ��ȯ
}