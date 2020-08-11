#include "data.h"
#include "utils.h"
#include "image.h"
#include "cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

list *get_paths(char *filename)
{
    char *path;
    //FILE *file = fopen("E:\\CNN_workspace\\yolo\\darknet-master\\build\\darknet\\x64\\data\\coco.names"/*filename*/, "r");
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

/*
char **get_random_paths_indexes(char **paths, int n, int m, int *indexes)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    for(i = 0; i < n; ++i){
        int index = random_gen()%m;
        indexes[i] = index;
        random_paths[i] = paths[index];
        if(i == 0) printf("%s\n", paths[index]);
    }
    pthread_mutex_unlock(&mutex);
    return random_paths;
}
*/

char **get_random_paths(char **paths, int n, int m)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    //printf("n = %d \n", n);
    for(i = 0; i < n; ++i){
        do {
            int index = random_gen() % m;
            random_paths[i] = paths[index];
            //if(i == 0) printf("%s\n", paths[index]);
            //printf("grp: %s\n", paths[index]);
            if (strlen(random_paths[i]) <= 4) printf(" Very small path to the image: %s \n", random_paths[i]);
        } while (strlen(random_paths[i]) == 0);
    }
    pthread_mutex_unlock(&mutex);
    return random_paths;
}

char **find_replace_paths(char **paths, int n, char *find, char *replace)
{
    char **replace_paths = calloc(n, sizeof(char*));
    int i;
    for(i = 0; i < n; ++i){
        char replaced[4096];
        find_replace(paths[i], find, replace, replaced);
        replace_paths[i] = copy_string(replaced);
    }
    return replace_paths;
}

matrix load_image_paths_gray(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image(paths[i], w, h, 3);

        image gray = grayscale_image(im);
        free_image(im);
        im = gray;

        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_paths(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], w, h);
        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_augment_paths(char **paths, int n, int use_flip, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image crop = random_augment_image(im, angle, aspect, min, max, size);
        int flip = use_flip ? random_gen() % 2 : 0;
        if (flip)
            flip_image(crop);
        random_distort_image(crop, hue, saturation, exposure);

        /*
        show_image(im, "orig");
        show_image(crop, "crop");
        cvWaitKey(0);
        */
        free_image(im);
        X.vals[i] = crop.data;
        X.cols = crop.h*crop.w*crop.c;
    }
    return X;
}

//用labels文档的路径将所有的ground truth boxes信息存入一个box_label结构体的数组中
box_label *read_boxes(char *filename, int *n)
{
	//char* aa = "E:\\CNN_workspace\\yolo\\darknet-master\\car_train_diganSet\\cfg\\labels\\2019-07-17-1563327575000-12-03-00-04-84-df_1004493547.txt";
	//filename = aa;
	box_label *boxes = calloc(1, sizeof(box_label));
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Can't open label file. (This can be normal only if you use MSCOCO) \n");
        //file_error(filename);
        FILE* fw = fopen("bad.list", "a");
        fwrite(filename, sizeof(char), strlen(filename), fw);
        char *new_line = "\n";
        fwrite(new_line, sizeof(char), strlen(new_line), fw);
        fclose(fw);

        *n = 0;
        return boxes;
    }
    float x, y, h, w;
    int id;
	int occFlag = -1;
    int count = 0;
	int res = 0;
#ifdef OPEN_OCC_CLASS_FLAG
	while (fscanf(file, "%d %f %f %f %f %d", &id, &x, &y, &w, &h, &occFlag) == 6)
#else
    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5)
#endif
	{
        boxes = realloc(boxes, (count+1)*sizeof(box_label));
#ifdef OPEN_OCC_CLASS_FLAG
		//0是车头，1 是车位，2 是不确定
		boxes[count].occFlag = occFlag;
#else
		boxes[count].occFlag = 3;
#endif
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;
        ++count;
    }
    fclose(file);
    *n = count;
	//if (count == 0)
	//{
	//	printf("\n\n%s\n\n", filename);
	//}
    return boxes;
}

void randomize_boxes(box_label *b, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        box_label swap = b[i];
        int index = random_gen()%n;
        b[i] = b[index];
        b[index] = swap;
    }
}

void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip)
{
    int i;
	int id;
    for(i = 0; i < n; ++i)
	{
		id = boxes[i].id;//获取当前box的类别
        if(boxes[i].x == 0 && boxes[i].y == 0) 
		{
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        boxes[i].left   = boxes[i].left  * sx - dx;
        boxes[i].right  = boxes[i].right * sx - dx;
        boxes[i].top    = boxes[i].top   * sy - dy;
        boxes[i].bottom = boxes[i].bottom* sy - dy;

		//在车检二分类问题中，如果下边缘超出了图像范围，box是0类，就用更改为1类
#ifdef OPEN_OCC_CLASS_FLAG
#if 0
		//printf("\n  n = %d, i = %d, boxes[i].bottom = %f", n, i, boxes[i].bottom);
		if (id == 0 && boxes[i].bottom >=0.9999)
		{
			boxes[i].id = 1;
			boxes[i].occFlag = 1;
		}
#endif
		//if (id == 0 && boxes[i].bottom >= 0.9999)
		//{
		//	boxes[i].occFlag = 1;
		//}
#endif
        if(flip)
		{
            float swap = boxes[i].left;
            boxes[i].left = 1. - boxes[i].right;
            boxes[i].right = 1. - swap;
        }

        boxes[i].left =  constrain(0, 1, boxes[i].left);
        boxes[i].right = constrain(0, 1, boxes[i].right);
        boxes[i].top =   constrain(0, 1, boxes[i].top);
        boxes[i].bottom =   constrain(0, 1, boxes[i].bottom);

        boxes[i].x = (boxes[i].left+boxes[i].right)/2;
        boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);

        boxes[i].w = constrain(0, 1, boxes[i].w);
        boxes[i].h = constrain(0, 1, boxes[i].h);
    }
}

void fill_truth_swag(char *path, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    replace_image_to_label(path, labelpath);

    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count && i < 30; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .0 || h < .0) continue;

        int index = (4+classes) * i;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;

        if (id < classes) truth[index+id] = 1;
    }
    free(boxes);
}

void fill_truth_region(char *path, float *truth, int classes, int num_boxes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    replace_image_to_label(path, labelpath);

    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .001 || h < .001) continue;

        int col = (int)(x*num_boxes);
        int row = (int)(y*num_boxes);

        x = x*num_boxes - col;
        y = y*num_boxes - row;

        int index = (col+row*num_boxes)*(5+classes);
        if (truth[index]) continue;
        truth[index++] = 1;

        if (id < classes) truth[index+id] = 1;
        index += classes;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;
    }
    free(boxes);
}
typedef struct line {
	int start;
	int end;
}myList;
//最多10个目标
int judgeLineInRanges(myList* segList, int count, int * expandSide)
{
	for (int j = 0; j < count; j++)
	{
		if (*expandSide > segList[j].start && *expandSide < segList[j].end)
		{
			return 1;
		}
		return 0;
	}
}
void adjust_jitter_base_GT_size(char *path, int* pleft, int* ptop, int* pright, int* pbottom, int img_w, int img_h)
{
	char labelpath[4096];
	replace_image_to_label(path, labelpath);

	int count = 0;
	int i;
	box_label *boxes = read_boxes(labelpath, &count);

	//获得了所有的标记box
	//查看切断的边界线，是否穿过box，穿过的话就让他避开
	int iLeft = 0, iTop = 0, iRight = 0, iBottom = 0;
	myList* segList_w = (myList*)malloc(count * sizeof(myList));
	myList* segList_h = (myList*)malloc(count * sizeof(myList));
	memset(segList_w, 0, sizeof(myList) * count);
	memset(segList_h, 0, sizeof(myList) * count);

	for (i = 0; i < count; i++)
	{
		iLeft = boxes[i].left;
		iTop = boxes[i].top;
		iRight = boxes[i].right;
		iBottom = boxes[i].bottom;

		segList_w[i].start = boxes[i].left * img_w;
		segList_w[i].end = boxes[i].right * img_w;


		segList_h[i].start = boxes[i].top * img_h;
		segList_h[i].end = boxes[i].bottom * img_h;
	}

	//遍历box数目，分别向两侧调整 TOP LEFT BOTTOM RIGHT以5个数为单位快速计算
	//printf("\n %s", path);
	//printf("\n left = %d, top = %d, right = %d, bot = %d", *pleft, *ptop, *pright, *pbottom);

	while (judgeLineInRanges(segList_w, count, pleft))
	{
		*pleft = (*pleft) - 5;
	}
	while (judgeLineInRanges(segList_h, count, ptop))
	{
		*ptop = (*ptop) - 5;
	}
	while (judgeLineInRanges(segList_w, count, pright))
	{
		*pright = (*pright) - 5;
	}
	while (judgeLineInRanges(segList_h, count, pbottom))
	{
		*pbottom = (*pbottom) - 5;
	}
	//printf("\n left = %d, top = %d, right = %d, bot = %d", *pleft, *ptop, *pright, *pbottom);

	free(boxes);
	free(segList_w);
	free(segList_h);
}

void patchAugmentation_image(char *path, int num_boxes, float *truth, int classes, int flip, float dx, float dy, float sx, float sy, int small_object, int net_w, int net_h/*, IplImage *src*/)
{
	//cv::Mat srcImg = Mat(src);
	//获得标签文件所在路径
	
	char labelpath[4096];
	replace_image_to_label(path, labelpath);

	int count = 0;
	int i;
	box_label *boxes = read_boxes(labelpath, &count);
	float lowest_w = 1.F / net_w;
	float lowest_h = 1.F / net_h;
	if (small_object == 1) {
		for (i = 0; i < count; ++i) {
			if (boxes[i].w < lowest_w) boxes[i].w = lowest_w;
			if (boxes[i].h < lowest_h) boxes[i].h = lowest_h;
		}
	}
	randomize_boxes(boxes, count);
	//correct_boxes根据图片调整比例，调整标注框的大小。
	correct_boxes(boxes, count, dx, dy, sx, sy, flip);
	if (count > num_boxes) count = num_boxes;
	float x, y, w, h;
	int id;
	int occ;
	int sub = 0;

	for (i = 0; i < count; ++i) {
		x = boxes[i].x;
		y = boxes[i].y;
		w = boxes[i].w;
		h = boxes[i].h;
		id = boxes[i].id;
#ifdef OPEN_OCC_CLASS_FLAG
		occ = boxes[i].occFlag;
#endif

		// not detect small objects
		//if ((w < 0.001F || h < 0.001F)) continue;
		// if truth (box for object) is smaller than 1x1 pix
		char buff[256];
		if (id >= classes) {
			printf("\n Wrong annotation: class_id = %d. But class_id should be [from 0 to %d] \n", id, classes);
			sprintf(buff, "echo %s \"Wrong annotation: class_id = %d. But class_id should be [from 0 to %d]\" >> bad_label.list", labelpath, id, classes);
			system(buff);
			getchar();
			++sub;
			continue;
		}
		if ((w < lowest_w || h < lowest_h)) {
			//sprintf(buff, "echo %s \"Very small object: w < lowest_w OR h < lowest_h\" >> bad_label.list", labelpath);
			//system(buff);
			++sub;
			continue;
		}
		if (x == 999999 || y == 999999) {
			printf("\n Wrong annotation: x = 0, y = 0 \n");
			sprintf(buff, "echo %s \"Wrong annotation: x = 0 or y = 0\" >> bad_label.list", labelpath);
			system(buff);
			++sub;
			continue;
		}
		if (x <= 0 || x > 1 || y <= 0 || y > 1) {
			printf("\n Wrong annotation: x = %f, y = %f \n", x, y);
			sprintf(buff, "echo %s \"Wrong annotation: x = %f, y = %f\" >> bad_label.list", labelpath, x, y);
			system(buff);
			++sub;
			continue;
		}
		if (w > 1) {
			printf("\n Wrong annotation: w = %f \n", w);
			sprintf(buff, "echo %s \"Wrong annotation: w = %f\" >> bad_label.list", labelpath, w);
			system(buff);
			w = 1;
		}
		if (h > 1) {
			printf("\n Wrong annotation: h = %f \n", h);
			sprintf(buff, "echo %s \"Wrong annotation: h = %f\" >> bad_label.list", labelpath, h);
			system(buff);
			h = 1;
		}
		if (x == 0) x += lowest_w;
		if (y == 0) y += lowest_h;

		//ground truth的赋值

		//occ = 3;
#ifdef OPEN_OCC_CLASS_FLAG
		truth[(i - sub) * 6 + 0] = x;
		truth[(i - sub) * 6 + 1] = y;
		truth[(i - sub) * 6 + 2] = w;
		truth[(i - sub) * 6 + 3] = h;
		truth[(i - sub) * 6 + 4] = id;
		truth[(i - sub) * 6 + 5] = occ;
#else
		truth[(i - sub) * 5 + 0] = x;
		truth[(i - sub) * 5 + 1] = y;
		truth[(i - sub) * 5 + 2] = w;
		truth[(i - sub) * 5 + 3] = h;
		truth[(i - sub) * 5 + 4] = id;
#endif

	}
	free(boxes);
}
//yolo根据图片路径读取标签文档功能 GroundTruth 加载函数
void fill_truth_detection(char *path, int num_boxes, float *truth, int classes, int flip, float dx, float dy, float sx, float sy,
    int small_object, int net_w, int net_h)
{
    char labelpath[4096];
    replace_image_to_label(path, labelpath);

    int count = 0;
    int i;
    box_label *boxes = read_boxes(labelpath, &count);
    float lowest_w = 1.F / net_w;
    float lowest_h = 1.F / net_h;
    if (small_object == 1) {
        for (i = 0; i < count; ++i) {
            if (boxes[i].w < lowest_w) boxes[i].w = lowest_w;
            if (boxes[i].h < lowest_h) boxes[i].h = lowest_h;
        }
    }
    randomize_boxes(boxes, count);
	//correct_boxes根据图片调整比例，调整标注框的大小。
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    if (count > num_boxes) count = num_boxes;
    float x, y, w, h;
    int id;
	int occ;
    int sub = 0;

    for (i = 0; i < count; ++i) {
        x = boxes[i].x;
        y = boxes[i].y;
        w = boxes[i].w;
        h = boxes[i].h;
        id = boxes[i].id;
#ifdef OPEN_OCC_CLASS_FLAG
		occ = boxes[i].occFlag;
#endif

        // not detect small objects
        //if ((w < 0.001F || h < 0.001F)) continue;
        // if truth (box for object) is smaller than 1x1 pix
        char buff[256];
        if (id >= classes) {
            printf("\n Wrong annotation: class_id = %d. But class_id should be [from 0 to %d] \n", id, classes);
            sprintf(buff, "echo %s \"Wrong annotation: class_id = %d. But class_id should be [from 0 to %d]\" >> bad_label.list", labelpath, id, classes);
            system(buff);
            getchar();
            ++sub;
            continue;
        }
        if ((w < lowest_w || h < lowest_h)) {
            //sprintf(buff, "echo %s \"Very small object: w < lowest_w OR h < lowest_h\" >> bad_label.list", labelpath);
            //system(buff);
            ++sub;
            continue;
        }
        if (x == 999999 || y == 999999) {
            printf("\n Wrong annotation: x = 0, y = 0 \n");
            sprintf(buff, "echo %s \"Wrong annotation: x = 0 or y = 0\" >> bad_label.list", labelpath);
            system(buff);
            ++sub;
            continue;
        }
        if (x <= 0 || x > 1 || y <= 0 || y > 1) {
            printf("\n Wrong annotation: x = %f, y = %f \n", x, y);
            sprintf(buff, "echo %s \"Wrong annotation: x = %f, y = %f\" >> bad_label.list", labelpath, x, y);
            system(buff);
            ++sub;
            continue;
        }
        if (w > 1) {
            printf("\n Wrong annotation: w = %f \n", w);
            sprintf(buff, "echo %s \"Wrong annotation: w = %f\" >> bad_label.list", labelpath, w);
            system(buff);
            w = 1;
        }
        if (h > 1) {
            printf("\n Wrong annotation: h = %f \n", h);
            sprintf(buff, "echo %s \"Wrong annotation: h = %f\" >> bad_label.list", labelpath, h);
            system(buff);
            h = 1;
        }
        if (x == 0) x += lowest_w;
        if (y == 0) y += lowest_h;

		//ground truth的赋值

		//occ = 3;
#ifdef OPEN_OCC_CLASS_FLAG
		truth[(i - sub) * 6 + 0] = x;
		truth[(i - sub) * 6 + 1] = y;
		truth[(i - sub) * 6 + 2] = w;
		truth[(i - sub) * 6 + 3] = h;
		truth[(i - sub) * 6 + 4] = id;
		truth[(i - sub) * 6 + 5] = occ;
#else
		truth[(i-sub)*5+0] = x;
        truth[(i-sub)*5+1] = y;
        truth[(i-sub)*5+2] = w;
        truth[(i-sub)*5+3] = h;
        truth[(i-sub)*5+4] = id;
#endif

    }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    free(boxes);
}
//yolo根据图片路径读取标签文档功能 GroundTruth 加载函数
int fill_truth_detection_int(char *path, int num_boxes, float *truth, int classes, int flip, float dx, float dy, float sx, float sy,
	int small_object, int net_w, int net_h)
{
	char labelpath[4096];
	replace_image_to_label(path, labelpath);

	int count = 0;
	int i;
	box_label *boxes = read_boxes(labelpath, &count);
	float lowest_w = 1.F / net_w;
	float lowest_h = 1.F / net_h;
	if (small_object == 1) {
		for (i = 0; i < count; ++i) {
			if (boxes[i].w < lowest_w) boxes[i].w = lowest_w;
			if (boxes[i].h < lowest_h) boxes[i].h = lowest_h;
		}
	}
	randomize_boxes(boxes, count);
	//correct_boxes根据图片调整比例，调整标注框的大小。
	correct_boxes(boxes, count, dx, dy, sx, sy, flip);
	if (count > num_boxes) count = num_boxes;
	float x, y, w, h;
	int id;
	int occ;
	int sub = 0;

	for (i = 0; i < count; ++i) {
		x = boxes[i].x;
		y = boxes[i].y;
		w = boxes[i].w;
		h = boxes[i].h;
		id = boxes[i].id;
#ifdef OPEN_OCC_CLASS_FLAG
		occ = boxes[i].occFlag;
#endif

		// not detect small objects
		//if ((w < 0.001F || h < 0.001F)) continue;
		// if truth (box for object) is smaller than 1x1 pix
		char buff[256];
		if (id >= classes) {
			printf("\n Wrong annotation: class_id = %d. But class_id should be [from 0 to %d] \n", id, classes);
			sprintf(buff, "echo %s \"Wrong annotation: class_id = %d. But class_id should be [from 0 to %d]\" >> bad_label.list", labelpath, id, classes);
			system(buff);
			getchar();
			++sub;
			continue;
		}
		if ((w < lowest_w || h < lowest_h)) {
			//sprintf(buff, "echo %s \"Very small object: w < lowest_w OR h < lowest_h\" >> bad_label.list", labelpath);
			//system(buff);
			++sub;
			continue;
		}
		if (x == 999999 || y == 999999) {
			printf("\n Wrong annotation: x = 0, y = 0 \n");
			sprintf(buff, "echo %s \"Wrong annotation: x = 0 or y = 0\" >> bad_label.list", labelpath);
			system(buff);
			++sub;
			continue;
		}
		if (x <= 0 || x > 1 || y <= 0 || y > 1) {
			printf("\n Wrong annotation: x = %f, y = %f \n", x, y);
			sprintf(buff, "echo %s \"Wrong annotation: x = %f, y = %f\" >> bad_label.list", labelpath, x, y);
			system(buff);
			++sub;
			continue;
		}
		if (w > 1) {
			printf("\n Wrong annotation: w = %f \n", w);
			sprintf(buff, "echo %s \"Wrong annotation: w = %f\" >> bad_label.list", labelpath, w);
			system(buff);
			w = 1;
		}
		if (h > 1) {
			printf("\n Wrong annotation: h = %f \n", h);
			sprintf(buff, "echo %s \"Wrong annotation: h = %f\" >> bad_label.list", labelpath, h);
			system(buff);
			h = 1;
		}
		if (x == 0) x += lowest_w;
		if (y == 0) y += lowest_h;

		//ground truth的赋值

		//occ = 3;
#ifdef OPEN_OCC_CLASS_FLAG
		truth[(i - sub) * 6 + 0] = x;
		truth[(i - sub) * 6 + 1] = y;
		truth[(i - sub) * 6 + 2] = w;
		truth[(i - sub) * 6 + 3] = h;
		truth[(i - sub) * 6 + 4] = id;
		truth[(i - sub) * 6 + 5] = occ;
#else
		truth[(i - sub) * 5 + 0] = x;
		truth[(i - sub) * 5 + 1] = y;
		truth[(i - sub) * 5 + 2] = w;
		truth[(i - sub) * 5 + 3] = h;
		truth[(i - sub) * 5 + 4] = id;
#endif


	}
	//setPatchToBigImage(pPatch, pBackground, boxes, count);
	free(boxes);
	return count;
}
#define NUMCHARS 37

void print_letters(float *pred, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        int index = max_index(pred+i*NUMCHARS, NUMCHARS);
        printf("%c", int_to_alphanum(index));
    }
    printf("\n");
}

void fill_truth_captcha(char *path, int n, float *truth)
{
    char *begin = strrchr(path, '/');
    ++begin;
    int i;
    for(i = 0; i < strlen(begin) && i < n && begin[i] != '.'; ++i){
        int index = alphanum_to_int(begin[i]);
        if(index > 35) printf("Bad %c\n", begin[i]);
        truth[i*NUMCHARS+index] = 1;
    }
    for(;i < n; ++i){
        truth[i*NUMCHARS + NUMCHARS-1] = 1;
    }
}

data load_data_captcha(char **paths, int n, int m, int k, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = make_matrix(n, k*NUMCHARS);
    int i;
    for(i = 0; i < n; ++i){
        fill_truth_captcha(paths[i], k, d.y.vals[i]);
    }
    if(m) free(paths);
    return d;
}

data load_data_captcha_encode(char **paths, int n, int m, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.X.cols = 17100;
    d.y = d.X;
    if(m) free(paths);
    return d;
}

void fill_truth(char *path, char **labels, int k, float *truth)
{
    int i;
    memset(truth, 0, k*sizeof(float));
    int count = 0;
    for(i = 0; i < k; ++i){
        if(strstr(path, labels[i])){
            truth[i] = 1;
            ++count;
        }
    }
    if(count != 1) printf("Too many or too few labels: %d, %s\n", count, path);
}

void fill_hierarchy(float *truth, int k, tree *hierarchy)
{
    int j;
    for(j = 0; j < k; ++j){
        if(truth[j]){
            int parent = hierarchy->parent[j];
            while(parent >= 0){
                truth[parent] = 1;
                parent = hierarchy->parent[parent];
            }
        }
    }
    int i;
    int count = 0;
    for(j = 0; j < hierarchy->groups; ++j){
        //printf("%d\n", count);
        int mask = 1;
        for(i = 0; i < hierarchy->group_size[j]; ++i){
            if(truth[count + i]){
                mask = 0;
                break;
            }
        }
        if (mask) {
            for(i = 0; i < hierarchy->group_size[j]; ++i){
                truth[count + i] = SECRET_NUM;
            }
        }
        count += hierarchy->group_size[j];
    }
}

matrix load_labels_paths(char **paths, int n, char **labels, int k, tree *hierarchy)
{
    matrix y = make_matrix(n, k);
    int i;
    for(i = 0; i < n && labels; ++i){
        fill_truth(paths[i], labels, k, y.vals[i]);
        if(hierarchy){
            fill_hierarchy(y.vals[i], k, hierarchy);
        }
    }
    return y;
}

matrix load_tags_paths(char **paths, int n, int k)
{
    matrix y = make_matrix(n, k);
    int i;
    int count = 0;
    for(i = 0; i < n; ++i){
        char label[4096];
        find_replace(paths[i], "imgs", "labels", label);
        find_replace(label, "_iconl.jpeg", ".txt", label);
        FILE *file = fopen(label, "r");
        if(!file){
            find_replace(label, "labels", "labels2", label);
            file = fopen(label, "r");
            if(!file) continue;
        }
        ++count;
        int tag;
        while(fscanf(file, "%d", &tag) == 1){
            if(tag < k){
                y.vals[i][tag] = 1;
            }
        }
        fclose(file);
    }
    printf("%d/%d\n", count, n);
    return y;
}

char **get_labels_custom(char *filename, int *size)
{
    list *plist = get_paths(filename);
    if(size) *size = plist->size;
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}

char **get_labels(char *filename)
{
    return get_labels_custom(filename, NULL);
}

void free_data(data d)
{
    if(!d.shallow){
        free_matrix(d.X);
        free_matrix(d.y);
    }else{
        free(d.X.vals);
        free(d.y.vals);
    }
}

data load_data_region(int n, char **paths, int m, int w, int h, int size, int classes, float jitter, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;


    int k = size*size*(5+classes);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);

        int oh = orig.h;
        int ow = orig.w;

        int dw = (ow*jitter);
        int dh = (oh*jitter);

        int pleft  = rand_uniform(-dw, dw);
        int pright = rand_uniform(-dw, dw);
        int ptop   = rand_uniform(-dh, dh);
        int pbot   = rand_uniform(-dh, dh);

        int swidth =  ow - pleft - pright;
        int sheight = oh - ptop - pbot;

        float sx = (float)swidth  / ow;
        float sy = (float)sheight / oh;

        int flip = random_gen()%2;
        image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

        float dx = ((float)pleft/ow)/sx;
        float dy = ((float)ptop /oh)/sy;

        image sized = resize_image(cropped, w, h);
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;

        fill_truth_region(random_paths[i], d.y.vals[i], classes, size, flip, dx, dy, 1./sx, 1./sy);

        free_image(orig);
        free_image(cropped);
    }
    free(random_paths);
    return d;
}

data load_data_compare(int n, char **paths, int m, int classes, int w, int h)
{
    if(m) paths = get_random_paths(paths, 2*n, m);
    int i,j;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*6;

    int k = 2*(classes);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i){
        image im1 = load_image_color(paths[i*2],   w, h);
        image im2 = load_image_color(paths[i*2+1], w, h);

        d.X.vals[i] = calloc(d.X.cols, sizeof(float));
        memcpy(d.X.vals[i],         im1.data, h*w*3*sizeof(float));
        memcpy(d.X.vals[i] + h*w*3, im2.data, h*w*3*sizeof(float));

        int id;
        float iou;

        char imlabel1[4096];
        char imlabel2[4096];
        find_replace(paths[i*2],   "imgs", "labels", imlabel1);
        find_replace(imlabel1, "jpg", "txt", imlabel1);
        FILE *fp1 = fopen(imlabel1, "r");

        while(fscanf(fp1, "%d %f", &id, &iou) == 2){
            if (d.y.vals[i][2*id] < iou) d.y.vals[i][2*id] = iou;
        }

        find_replace(paths[i*2+1], "imgs", "labels", imlabel2);
        find_replace(imlabel2, "jpg", "txt", imlabel2);
        FILE *fp2 = fopen(imlabel2, "r");

        while(fscanf(fp2, "%d %f", &id, &iou) == 2){
            if (d.y.vals[i][2*id + 1] < iou) d.y.vals[i][2*id + 1] = iou;
        }

        for (j = 0; j < classes; ++j){
            if (d.y.vals[i][2*j] > .5 &&  d.y.vals[i][2*j+1] < .5){
                d.y.vals[i][2*j] = 1;
                d.y.vals[i][2*j+1] = 0;
            } else if (d.y.vals[i][2*j] < .5 &&  d.y.vals[i][2*j+1] > .5){
                d.y.vals[i][2*j] = 0;
                d.y.vals[i][2*j+1] = 1;
            } else {
                d.y.vals[i][2*j]   = SECRET_NUM;
                d.y.vals[i][2*j+1] = SECRET_NUM;
            }
        }
        fclose(fp1);
        fclose(fp2);

        free_image(im1);
        free_image(im2);
    }
    if(m) free(paths);
    return d;
}

data load_data_swag(char **paths, int n, int classes, float jitter)
{
    int index = random_gen()%n;
    char *random_path = paths[index];

    image orig = load_image_color(random_path, 0, 0);
    int h = orig.h;
    int w = orig.w;

    data d = {0};
    d.shallow = 0;
    d.w = w;
    d.h = h;

    d.X.rows = 1;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    int k = (4+classes)*30;
    d.y = make_matrix(1, k);

    int dw = w*jitter;
    int dh = h*jitter;

    int pleft  = rand_uniform(-dw, dw);
    int pright = rand_uniform(-dw, dw);
    int ptop   = rand_uniform(-dh, dh);
    int pbot   = rand_uniform(-dh, dh);

    int swidth =  w - pleft - pright;
    int sheight = h - ptop - pbot;

    float sx = (float)swidth  / w;
    float sy = (float)sheight / h;

    int flip = random_gen()%2;
    image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

    float dx = ((float)pleft/w)/sx;
    float dy = ((float)ptop /h)/sy;

    image sized = resize_image(cropped, w, h);
    if(flip) flip_image(sized);
    d.X.vals[0] = sized.data;

    fill_truth_swag(random_path, d.y.vals[0], classes, flip, dx, dy, 1./sx, 1./sy);

    free_image(orig);
    free_image(cropped);

    return d;
}

#ifdef OPENCV

#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/version.hpp"

#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio_c.h"
#include "opencv2/imgcodecs/imgcodecs_c.h"
#endif	//CV_VERSION_EPOCH

#include "http_stream.h"


#ifdef WIN32
#define ICE_min __min
#define ICE_max __max
#else
#define ICE_min(a, b) ((a) > (b) ? (b) : (a))
#define ICE_max(a, b) ((a) > (b) ? (a) : (b))
#endif

//!< 将两个图像拼接到一起，side=0，是左侧车位，side=1，是右侧车位
CvRect mergeImages(IplImage* pPatch, IplImage* pBackground, box_label box, int side)
{
	//1.0 根据是哪一侧，计算贴图的位置
	int left = box.left * pBackground->width;
	int right = box.right * pBackground->width;
	int top = box.top * pBackground->height;
	int botm = box.bottom * pBackground->height;

	int boxHeight = botm - top + 1;


	float hole_left = 0.;
	float hole_right = 0.;
	float hole_top = 0.;
	float hole_bottom = 0.;

	int margin = 0;
	if (side == 0)//左侧车位，向右贴图
	{
		hole_left = right - (right - left) * 0.35;//patch所贴位置的左边缘
		//距图像右边缘距离
		margin = pBackground->width - hole_left - 1;

		hole_right = hole_left + margin;
	}
	else//右侧车位，向左贴图
	{
		hole_right = left + (right - left) * 0.35;//patch所贴位置的右边缘

		margin = hole_right;
		hole_left = 0;

	}
	//patch的高度在box高度和图像高度之间动态变化。
	int patchHeight = rand_uniform_strong(boxHeight * 0.85, pBackground->height - boxHeight*0.1);
	float patchRatio = pPatch->width * 1.0 / pPatch->height;
	int patchWidth = patchHeight * patchRatio;
	//以上得到了patch所放置位置的宽和高
	//现在开始计算具体的位置
	int patch_x = 0;
	int patch_y = 0;
	if (side == 0)//左侧车位，调整右边界左移-----左边界不能动
	{
		patch_x = hole_left;
		if (patch_x + patchWidth >= pBackground->width)
		{
			int tmp = patch_x + patchWidth - pBackground->width + 1;
			patchWidth -= tmp;
		}
	}
	else
	{
		patch_x = hole_right - patchWidth;
		if (patch_x < 0 )
		{
			int tmp = abs(patch_x);
			patch_x = 0;
			patchWidth = patchWidth - tmp - 1;
		}
	}


	//以上步骤得到了可以放patch的最大范围
	//step2.0 根据hole_left hole_right hole_top hole_bottom 进行贴图操作
	CvRect rect;
	rect.x = patch_x;
	rect.y = (pBackground->height - patchHeight)/2;
	rect.width = patchWidth;
	rect.height = patchHeight;

	return rect;
}
//!< 将两个图像拼接到一起，side=0，是左侧车位，side=1，是右侧车位
//!< 将patch放到车的边上，防止发生大目标合并的风险
CvRect mergeImages_put_patch_side(IplImage* pPatch, IplImage* pBackground, box_label box, int side)
{
	//1.0 根据是哪一侧，计算贴图的位置
	int left = box.left * pBackground->width;
	int right = box.right * pBackground->width;
	int top = box.top * pBackground->height;
	int botm = box.bottom * pBackground->height;

	int boxHeight = botm - top + 1;


	float hole_left = 0.;
	float hole_right = 0.;
	float hole_top = 0.;
	float hole_bottom = 0.;

	int margin = 0;
	if (side == 0)//左侧车位，向右贴图
	{
		hole_left = right + (right - left) * 0.01;//patch所贴位置的左边缘
		hole_left = ICE_min(hole_left, pBackground->width - 5);
		//距图像右边缘距离
		margin = pBackground->width - hole_left - 1;

		hole_right = hole_left + margin;
		if (hole_right < hole_left)//说明超出图像范围了
		{
			//hole_right = 
			hole_right = hole_left + 1;
		}
	}
	else//右侧车位，向左贴图
	{
		hole_right = left - (right - left) * 0.01;//patch所贴位置的右边缘

		margin = hole_right;
		hole_left = 0;
		if (hole_right < 0)
		{
			hole_right = 1;
		}

	}
	if (hole_right >= pBackground->width - 1)
	{
		hole_right = pBackground->width - 5;
		hole_left = hole_right - 5;
	}
	if (hole_left <=0)
	{
		hole_left = 2;
		hole_right = 7;

	}
	//patch的高度在box高度和图像高度之间动态变化。
	int patchHeight = rand_uniform_strong(boxHeight * 0.85, pBackground->height - boxHeight*0.1);
	float patchRatio = pPatch->width * 1.0 / pPatch->height;
	int patchWidth = patchHeight * patchRatio;
	//以上得到了patch所放置位置的宽和高
	//现在开始计算具体的位置
	int patch_x = 0;
	int patch_y = 0;
	if (side == 0)//左侧车位，调整右边界左移-----左边界不能动
	{
		patch_x = hole_left;
		if (patch_x + patchWidth >= pBackground->width)
		{
			int tmp = patch_x + patchWidth - pBackground->width + 2;
			patchWidth -= tmp;
		}
	}
	else
	{
		patch_x = hole_right - patchWidth;
		if (patch_x < 0 )
		{
			int tmp = abs(patch_x);
			patch_x = 0;
			patchWidth = patchWidth - tmp - 1;
		}
	}
	if (patchWidth<=2)
	{
		patchWidth = 2;
	}

	//以上步骤得到了可以放patch的最大范围
	//step2.0 根据hole_left hole_right hole_top hole_bottom 进行贴图操作
	CvRect rect;
	rect.x = patch_x;
	rect.y = (pBackground->height - patchHeight)/2;
	rect.width = patchWidth;
	rect.height = patchHeight;

	return rect;
}
//!< 将pPatch 复制到 pBackground的 pos位置的地方
//!< pPatch 补丁图像
//!< pBackground 原始输入待训练样本图像
//!< pos patch需要叠加到的地方，这个框是标记框的一半车尾大小

int setPatchToBigImage(IplImage* pPatch, IplImage* pBackground, float *truth, int count, int patch_pos_sidebyside_flag)
{
	//box_label *boxes = read_boxes(labelpath, &count);
	//输入的两幅图像都是原始图
	//step1.0 找到最宽的车检label索引
	float largestWidth = 0;
	int largestIndex = -1;

	int i;
	int sub = 0;
	float x, y, w, h;
	box_label* boxes = calloc(count, sizeof(box_label));
	for (i = 0; i < count; ++i)
	{
		//occ = 3;
#ifdef OPEN_OCC_CLASS_FLAG
		x = truth[(i) * 6 + 0];
		y = truth[(i) * 6 + 1];
		w = truth[(i) * 6 + 2];
		h = truth[(i) * 6 + 3];
#else
		x = truth[(i) * 5 + 0];
		y = truth[(i) * 5 + 1];
		w = truth[(i) * 5 + 2];
		h = truth[(i) * 5 + 3];
#endif

		boxes[i].x = x;
		boxes[i].y = y;
		boxes[i].w = w;
		boxes[i].h = h;
		boxes[i].x = x;

		boxes[i].left = x - w / 2;
		boxes[i].right = x + w / 2;
		boxes[i].top = y - h / 2;
		boxes[i].bottom = y + h / 2;
		if (w > largestWidth)
		{
			largestIndex = i;
			largestWidth = w;
		}
	}


	if (largestIndex < 0)
	{
		return 0;
	}
	//step2.0 通过剩下box和最大box左右边界的相对关系，判定最可能是左侧还是右侧车位
	bool bAllRight = true;
	bool bAllLeft = true;
	for (i = 0; i < count; i++)
	{
		if (i == largestIndex)//跳过最大的框
		{
			continue;
		}
		if (boxes[i].right > boxes[largestIndex].right)//非左侧车位
		{
			bAllLeft = false;
		}
		if (boxes[i].left < boxes[largestIndex].left)//非右侧车位
		{
			bAllRight = false;
		}

	}
	if ((!bAllRight) && (!bAllLeft) || (bAllRight && bAllLeft))//无法判定是哪一侧的车位
	{
		return 0;
	}
	//step3.0 确定了是哪一测的车位，接下来，开始读取patch图片，贴到车身上。
	CvSize size;
	CvRect rect;
	if (bAllRight && (!bAllLeft))//右侧车位
	{
		if (patch_pos_sidebyside_flag)
		{
			rect = mergeImages_put_patch_side(pPatch, pBackground, boxes[largestIndex], 1);
		}
		else
		{
			rect = mergeImages(pPatch, pBackground, boxes[largestIndex], 1);
		}
	}
	else if ((!bAllRight) && bAllLeft)//左侧车位
	{
		if (patch_pos_sidebyside_flag)
		{
			rect = mergeImages_put_patch_side(pPatch, pBackground, boxes[largestIndex], 0);
		}
		else
		{
			rect = mergeImages(pPatch, pBackground, boxes[largestIndex], 0);
		}
	}
	//安全检查
	if (rect.width < 2) rect.width = 2;
	if (rect.height < 2) rect.height = 2;

	if (rect.x < 0) rect.x = 0;
	if (rect.y < 0) rect.x = 0;

	if (rect.x +rect.width >= pBackground->width)
	{
		int tmp = rect.x + rect.width - pBackground->width;
		rect.width = ICE_max(5, rect.width - tmp);
	}
	if (rect.y + rect.height >= pBackground->width)
	{
		int tmp = rect.y + rect.height - pBackground->width;
		rect.height = ICE_max(5, rect.height - tmp);

	}

	size.height = rect.height;
	size.width = rect.width;

	IplImage* resizedPatch = cvCreateImage(size, pPatch->depth, pPatch->nChannels);
	cvResize(pPatch, resizedPatch, CV_INTER_CUBIC);

	cvSetImageROI(pBackground, cvRect(rect.x, rect.y, resizedPatch->width, resizedPatch->height)); //设置背景图上的ROI

	cvCopy(resizedPatch, pBackground, 0);				//复制patch到ROI中
	cvResetImageROI(pBackground);						//关闭ROI索引

	//cvShowImage("dst", pBackground);
	//cvWatikey(0);
	cvNamedWindow("dst", 0);
	cvResizeWindow("dst", 960, 544);
	cvShowImage("dst", pBackground);
	cvWaitKey(0);

	cvReleaseImage(&resizedPatch);
	free(boxes);
}

int getPathFileName(char* src, char path[], char* name)
{
	int x = strlen(src);
	char ch = '\\';
	char*q = strrchr(src, ch) + 1;
	if (q == 0x0000000000000001 || q == NULL)
	{
		ch = '/';
		q = strrchr(src, ch) + 1;
	}
	int number = q - src - 1;
	//strncpy(path, src, number);
	//sprintf(path, "%s%c", path, '\0');
	//strncpy(name, q, x);
	return number;
}
#include <io.h>
FILE *fp;

int findJPGimgNum(char *to_search_)
{
	int num = -1;
	char findType[300] = { 0 };
	strcpy(findType, to_search_);
	sprintf(findType, "%s/*.jpg", findType);
	const char *to_search = findType;    //欲查找的文件，支持通配符

	long handle;                                                     //用于查找的句柄
	struct _finddata_t fileinfo;                                     //文件信息的结构体

	handle = _findfirst(to_search, &fileinfo);                          //第一次查找
	if (-1 == handle)
		return -1;
	while (!_findnext(handle, &fileinfo))                              //循环查找其他符合的文件，直到找不到其他的为止
	{
		if (num == -1)
		{
			num = 1;
		}
		//printf("\npositive/%s 0\r", fileinfo.name);
		num++;
	}
	_findclose(handle);                                              //关闭句柄

	printf("output done.\n");

	//system("pause");
	return num;
}

void Frosted_glass_Aug(IplImage* in, IplImage* out, int Number)
{
	int randomNum;
	CvScalar pixel_v;
	uchar* b_pixel;
	uchar* g_pixel;
	uchar* r_pixel;
	for (int i = 0; i < in->height - Number; i++)
	{
		for (int j = 0; j < in->width - Number; j++)
		{
			randomNum = random_gen() % Number;
			b_pixel = (uchar*)(out->imageData + i*out->widthStep + (j*out->nChannels + 0));
			g_pixel = (uchar*)(out->imageData + i*out->widthStep + (j*out->nChannels + 1));
			r_pixel = (uchar*)(out->imageData + i*out->widthStep + (j*out->nChannels + 2));

			*b_pixel = cvGet2D(in, i + randomNum, j + randomNum).val[0];
			*g_pixel = cvGet2D(in, i + randomNum, j + randomNum).val[1];
			*r_pixel = cvGet2D(in, i + randomNum, j + randomNum).val[2];
		}
	}

}

void blur_iplimage(IplImage* in, IplImage* out, int type)
{
	int kernalSize = random_gen() % 2 == 0 ? 3 : 5;
	if (type == 0)
	{
		Frosted_glass_Aug(in, out, kernalSize);//毛玻璃
	}
	else if (type == 1)
	{
		//AddGuassianNoise(in, out);
	}
	else if (type == 2)
	{
		cvSmooth(in, out, CV_BLUR, kernalSize, kernalSize, kernalSize, kernalSize);  //  简单平均
	}
	else if (type == 3)
	{
		cvSmooth(in, out, CV_MEDIAN, kernalSize, kernalSize, kernalSize, kernalSize);  //  中值滤波
	}
	else if (type == 4)
	{
		cvSmooth(in, out, CV_GAUSSIAN, kernalSize, kernalSize, kernalSize, kernalSize);  //  Gauss 平滑
	}
	else
	{
	}
}
data load_data_detection(int n, char **paths, int m, int w, int h, int c, int boxes, int classes, int use_flip, float jitter, float hue, float saturation, float exposure, int small_object)
{
	static int sampleCount = 0;
    c = c ? c : 3;
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;// 一个线程加载的样本数量为n个
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*c;//每一行都有（每个样本就是cols个float数据）
#ifdef OPEN_OCC_CLASS_FLAG
    d.y = make_matrix(n,( 5 + 1 )*boxes);//存放ID x, y, w, h , occFlag
#else
	d.y = make_matrix(n, 5*boxes);
#endif
    for(i = 0; i < n; ++i){
		//sampleCount++;
		//printf("\nsample = %d", i);
		const char *filename = random_paths[i];
		//char *filename_patch = random_paths[i];
		char filename_patch[1000] = { 0 };
		char filename_patch_2[1000] = { 0 };
		strcpy(filename_patch, filename);

        int flag = (c >= 3);
		IplImage *src_in;
		IplImage *src_out;
		IplImage *patch;
		//filename = "C:\\Users\\Administrator\\Desktop\\qqq\\testimg.jpg";

		//filename = "E:/CNN_workspace/yolo/darknet-master/car_train_4/data/car/IPGH/JPEGImages/16392000207_1004492141.jpg";
	
		if ((src_in = cvLoadImage(filename, flag)) == 0)
        {
            fprintf(stderr, "Cannot load image \"%s\"\n", filename);
            char buff[256];
            sprintf(buff, "echo %s >> bad.list", filename);
            system(buff);
            continue;
            //exit(0);
        }
		src_out = cvCloneImage(src_in);
		//----------------blur--------------------
		int blurType = random_gen() % 7;
		blur_iplimage(src_in, src_out, blurType);
		//cvNamedWindow("in", 0);
		//cvNamedWindow("out", 0);
		//cvResizeWindow("in", 960, 544);
		//cvResizeWindow("out", 960, 544);
		//cvShowImage("in", src_in);
		//cvShowImage("out", src_out);
		//cvWaitKey(0);
		//---------------------------------------
		//先找到标记文件，看其中的gt——box是怎样分布的，有以下几种情况
        int oh = src_in->height;//原始图像高度
        int ow = src_in->width;//原始图像宽度
		float jitterx = 0;
		float jittery = 0;

		float w2hRatio = (ow)*1.0 / oh;
        int dh = (oh * jitter);
        int dw = (ow * jitter);

        int pleft  = rand_uniform_strong(-dw, dw);
        int pright = rand_uniform_strong(-dw, dw);
        int ptop   = rand_uniform_strong(-dh, dh);
        int pbot   = rand_uniform_strong(-dh, dh);
		//int* pleft, int* ptop, int* pright, int* bottom, 
#ifdef ATTRI_TRAIN_LIMIT//如果是进行属性检测

		//在这里对四个偏移量进行调整，不能对目标进行截断，因为：如果截断了，那车头、车尾、不确定就没办法区分了
		adjust_jitter_base_GT_size(filename, &pleft, &ptop, &pright, &pbot, ow, oh);
#else

#endif
#define SELF_AUGMENTATION
#ifdef SELF_AUGMENTATION

		//char cCurFolder[2000];
		//_getcwd(cCurFolder, 1000);
		////printf("%s", cCurFolder);
		////detector test /cfg/IPGH80.data cfg\yolov3-tiny_oriSrc.cfg  \yolov3-tiny_80.weights E:\CNN_workspace\yolo\darknet-master\build\darknet\x64\dog.jpg
		//char PatchFolder[2000];//argv[3]
		//sprintf(PatchFolder, "%s\\patch\\", cCurFolder);
		//
		//1.0 获得直接替换的patch目录
		replace_image_to_image(filename_patch, filename_patch_2);
		
		//2.0 获得patch的最终文件名
		char patchPath[500];
		char patchName[100];

		int number = getPathFileName(filename_patch_2, &patchPath, patchName);
		strncpy(patchPath, filename_patch_2, number);

		static int interval = 0;
		if (sampleCount%640 == 0)//每5张图片，检查一次是否新增了负样本
		{
			interval = findJPGimgNum(patchPath);
		}
		//CvMat
		sampleCount++;
		if (sampleCount > 10000)
		{
			sampleCount = 0;
		}
		int iPatchNum = random_gen() % interval;
		sprintf(patchPath, "%s/_%d_.jpg", patchPath, iPatchNum);
		//char* patchPath_ = "E:/CNN_workspace/yolo/darknet-master/car_train_4/data/car/IPGH/patch/_156_.jpg";
		//char* patchPath_ = "E:\\公司资源\\爬图工具\\爬图工具\\download_images\\行人2/_0_.jpg";
		//printf("\n%s", filename); printf("\n%s", patchPath);
		if ((patch = cvLoadImage(patchPath, flag)) == 0)
		{
			fprintf(stderr, "Cannot load image \"%s\"\n", filename);
			char buff[256];
			sprintf(buff, "echo %s >> bad.list", filename);
			system(buff);
			continue;
			//exit(0);
		}
		else
		{
			//cvNamedWindow("patch", 0);
			//cvNamedWindow("src", 0);
			//cvNamedWindow("dst", 0);
			//cvShowImage("src", src);
			//cvShowImage("patch", patch);
			//3.0 读取图片后 开始增广

			int runRatio = random_gen() % 4 == 0;//25%的概率对图像进行增广
			if (runRatio)//
			{
				//获取标签文件中的box
				int count = fill_truth_detection_int(filename, boxes,d.y.vals[i], classes, 0, 0, 0, 1., 1. , small_object, w, h);
				//只有box数目大于2才能确定车辆位于相机的哪一侧
				if (count >=2)
				{
					//printf("\nbig = %s", filename);
					//printf("\npatch = %s", patchPath);
					int pos_side_by_side = random_gen()%5 > 2 ? 1 :0;
					//pos_side_by_side = 1;
					setPatchToBigImage(patch, src_out, d.y.vals[i], count, pos_side_by_side);

				}
			}

		}
		memset(patchPath, 0, 500);
		memset(patchName, 0, 100);

		cvReleaseImage(&patch);
#endif
        int swidth =  ow - pleft - pright;//抖动后的宽度
        int sheight = oh - ptop - pbot;//抖动后的高度

        float sx = (float)swidth  / ow;//抖动后对抖动前的 宽度、高度 比例
        float sy = (float)sheight / oh;

        int flip = use_flip ? random_gen()%2 : 0;

        float dx = ((float)pleft/ow)/sx;//偏移量/抖动后的宽度
        float dy = ((float)ptop /oh)/sy;//偏移量/抖动后的高度

        float dhue = rand_uniform_strong(-hue, hue);//亮度
        float dsat = rand_scale(saturation);
        float dexp = rand_scale_darkness(exposure);
		
        image ai = image_data_augmentation(src_out, w, h, pleft, ptop, swidth, sheight, flip, jitter, dhue, dsat, dexp);
        d.X.vals[i] = ai.data;

  //      show_image(ai, "aug");
		//static int iii = 0;
		//if (iii)
		//{
		//	IplImage* res = image_to_ipl(ai);
		//	cvShowImage("res", res);
		//	cvWaitKey(0);
		//}
  //      cvWaitKey(0);
		//GroundTruth 加载函数
        fill_truth_detection(filename, boxes, d.y.vals[i], classes, flip, dx, dy, 1./sx, 1./sy, small_object, w, h);

		cvReleaseImage(&src_in);
		cvReleaseImage(&src_out);
    }
    free(random_paths);
    return d;
}
#else    // OPENCV
data load_data_detection(int n, char **paths, int m, int w, int h, int c, int boxes, int classes, int use_flip, float jitter, float hue, float saturation, float exposure, int small_object)
{
    c = c ? c : 3;
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = { 0 };
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*c;

    d.y = make_matrix(n, 5 * boxes);
    for (i = 0; i < n; ++i) {
        image orig = load_image(random_paths[i], 0, 0, c);

        int oh = orig.h;
        int ow = orig.w;

        int dw = (ow*jitter);
        int dh = (oh*jitter);

        int pleft = rand_uniform_strong(-dw, dw);
        int pright = rand_uniform_strong(-dw, dw);
        int ptop = rand_uniform_strong(-dh, dh);
        int pbot = rand_uniform_strong(-dh, dh);

        int swidth = ow - pleft - pright;
        int sheight = oh - ptop - pbot;

        float sx = (float)swidth / ow;
        float sy = (float)sheight / oh;

        int flip = use_flip ? random_gen() % 2 : 0;
        image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

        float dx = ((float)pleft / ow) / sx;
        float dy = ((float)ptop / oh) / sy;

        image sized = resize_image(cropped, w, h);
        if (flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;

        fill_truth_detection(random_paths[i], boxes, d.y.vals[i], classes, flip, dx, dy, 1. / sx, 1. / sy, small_object, w, h);

        free_image(orig);
        free_image(cropped);
    }
    free(random_paths);
    return d;
}
#endif    // OPENCV

void *load_thread(void *ptr)
{
    //srand(time(0));
    //printf("Loading data: %d\n", random_gen());
    load_args a = *(struct load_args*)ptr;
    if(a.exposure == 0) a.exposure = 1;
    if(a.saturation == 0) a.saturation = 1;
    if(a.aspect == 0) a.aspect = 1;

    if (a.type == OLD_CLASSIFICATION_DATA){
        *a.d = load_data_old(a.paths, a.n, a.m, a.labels, a.classes, a.w, a.h);
    } else if (a.type == CLASSIFICATION_DATA){
        *a.d = load_data_augment(a.paths, a.n, a.m, a.labels, a.classes, a.hierarchy, a.flip, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    } else if (a.type == SUPER_DATA){
        *a.d = load_data_super(a.paths, a.n, a.m, a.w, a.h, a.scale);
    } else if (a.type == WRITING_DATA){
        *a.d = load_data_writing(a.paths, a.n, a.m, a.w, a.h, a.out_w, a.out_h);
    } else if (a.type == REGION_DATA){
        *a.d = load_data_region(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure);
    } else if (a.type == DETECTION_DATA){
        *a.d = load_data_detection(a.n, a.paths, a.m, a.w, a.h, a.c, a.num_boxes, a.classes, a.flip, a.jitter, a.hue, a.saturation, a.exposure, a.small_object);
    } else if (a.type == SWAG_DATA){
        *a.d = load_data_swag(a.paths, a.n, a.classes, a.jitter);
    } else if (a.type == COMPARE_DATA){
        *a.d = load_data_compare(a.n, a.paths, a.m, a.classes, a.w, a.h);
    } else if (a.type == IMAGE_DATA){
        *(a.im) = load_image(a.path, 0, 0, a.c);
        *(a.resized) = resize_image(*(a.im), a.w, a.h);
    }else if (a.type == LETTERBOX_DATA) {
        *(a.im) = load_image(a.path, 0, 0, a.c);
        *(a.resized) = letterbox_image(*(a.im), a.w, a.h);
    } else if (a.type == TAG_DATA){
        *a.d = load_data_tag(a.paths, a.n, a.m, a.classes, a.flip, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    }
    free(ptr);
    return 0;
}

pthread_t load_data_in_thread(load_args args)
{
    pthread_t thread;
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_thread, ptr)) error("Thread creation failed");
    return thread;
}

void *load_threads(void *ptr)
{
    //srand(time(0));
    int i;
    load_args args = *(load_args *)ptr;
    if (args.threads == 0) args.threads = 1;
    data *out = args.d;
    int total = args.n;
    free(ptr);
    data *buffers = calloc(args.threads, sizeof(data));
    pthread_t *threads = calloc(args.threads, sizeof(pthread_t));
    for(i = 0; i < args.threads; ++i){
        args.d = buffers + i;
        args.n = (i+1) * total/args.threads - i * total/args.threads;
        threads[i] = load_data_in_thread(args);
    }
    for(i = 0; i < args.threads; ++i){
        pthread_join(threads[i], 0);
    }
	//concat_datas的作用是将加载的数据整合到一起。最后释放资源。
    *out = concat_datas(buffers, args.threads);
    out->shallow = 0;
    for(i = 0; i < args.threads; ++i){
        buffers[i].shallow = 1;
        free_data(buffers[i]);
    }
    free(buffers);
    free(threads);
    return 0;
}

pthread_t load_data(load_args args)
{
    pthread_t thread;
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_threads, ptr)) error("Thread creation failed");
    return thread;
}

data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h)
{
    if(m) paths = get_random_paths(paths, n, m);
    char **replace_paths = find_replace_paths(paths, n, ".png", "-label.png");
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_image_paths_gray(replace_paths, n, out_w, out_h);
    if(m) free(paths);
    int i;
    for(i = 0; i < n; ++i) free(replace_paths[i]);
    free(replace_paths);
    return d;
}

data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_labels_paths(paths, n, labels, k, 0);
    if(m) free(paths);
    return d;
}

/*
   data load_data_study(char **paths, int n, int m, char **labels, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
   {
   data d = {0};
   d.indexes = calloc(n, sizeof(int));
   if(m) paths = get_random_paths_indexes(paths, n, m, d.indexes);
   d.shallow = 0;
   d.X = load_image_augment_paths(paths, n, flip, min, max, size, angle, aspect, hue, saturation, exposure);
   d.y = load_labels_paths(paths, n, labels, k);
   if(m) free(paths);
   return d;
   }
 */

data load_data_super(char **paths, int n, int m, int w, int h, int scale)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;

    int i;
    d.X.rows = n;
    d.X.vals = calloc(n, sizeof(float*));
    d.X.cols = w*h*3;

    d.y.rows = n;
    d.y.vals = calloc(n, sizeof(float*));
    d.y.cols = w*scale * h*scale * 3;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image crop = random_crop_image(im, w*scale, h*scale);
        int flip = random_gen()%2;
        if (flip) flip_image(crop);
        image resize = resize_image(crop, w, h);
        d.X.vals[i] = resize.data;
        d.y.vals[i] = crop.data;
        free_image(im);
    }

    if(m) free(paths);
    return d;
}

data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int use_flip, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_augment_paths(paths, n, use_flip, min, max, size, angle, aspect, hue, saturation, exposure);
    d.y = load_labels_paths(paths, n, labels, k, hierarchy);
    if(m) free(paths);
    return d;
}

data load_data_tag(char **paths, int n, int m, int k, int use_flip, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.w = size;
    d.h = size;
    d.shallow = 0;
    d.X = load_image_augment_paths(paths, n, use_flip, min, max, size, angle, aspect, hue, saturation, exposure);
    d.y = load_tags_paths(paths, n, k);
    if(m) free(paths);
    return d;
}

matrix concat_matrix(matrix m1, matrix m2)
{
    int i, count = 0;
    matrix m;
    m.cols = m1.cols;
    m.rows = m1.rows+m2.rows;
    m.vals = calloc(m1.rows + m2.rows, sizeof(float*));
    for(i = 0; i < m1.rows; ++i){
        m.vals[count++] = m1.vals[i];
    }
    for(i = 0; i < m2.rows; ++i){
        m.vals[count++] = m2.vals[i];
    }
    return m;
}

data concat_data(data d1, data d2)
{
    data d = {0};
    d.shallow = 1;
    d.X = concat_matrix(d1.X, d2.X);
    d.y = concat_matrix(d1.y, d2.y);
    return d;
}

data concat_datas(data *d, int n)
{
    int i;
    data out = {0};
    for(i = 0; i < n; ++i){
        data new = concat_data(d[i], out);
        free_data(out);
        out = new;
    }
    return out;
}

data load_categorical_data_csv(char *filename, int target, int k)
{
    data d = {0};
    d.shallow = 0;
    matrix X = csv_to_matrix(filename);
    float *truth_1d = pop_column(&X, target);
    float **truth = one_hot_encode(truth_1d, X.rows, k);
    matrix y;
    y.rows = X.rows;
    y.cols = k;
    y.vals = truth;
    d.X = X;
    d.y = y;
    free(truth_1d);
    return d;
}

data load_cifar10_data(char *filename)
{
    data d = {0};
    d.shallow = 0;
    long i,j;
    matrix X = make_matrix(10000, 3072);
    matrix y = make_matrix(10000, 10);
    d.X = X;
    d.y = y;

    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);
    for(i = 0; i < 10000; ++i){
        unsigned char bytes[3073];
        fread(bytes, 1, 3073, fp);
        int class_id = bytes[0];
        y.vals[i][class_id] = 1;
        for(j = 0; j < X.cols; ++j){
            X.vals[i][j] = (double)bytes[j+1];
        }
    }
    //translate_data_rows(d, -128);
    scale_data_rows(d, 1./255);
    //normalize_data_rows(d);
    fclose(fp);
    return d;
}

void get_random_batch(data d, int n, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = random_gen()%d.X.rows;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void get_next_batch(data d, int n, int offset, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = offset + j;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void smooth_data(data d)
{
    int i, j;
    float scale = 1. / d.y.cols;
    float eps = .1;
    for(i = 0; i < d.y.rows; ++i){
        for(j = 0; j < d.y.cols; ++j){
            d.y.vals[i][j] = eps * scale + (1-eps) * d.y.vals[i][j];
        }
    }
}

data load_all_cifar10()
{
    data d = {0};
    d.shallow = 0;
    int i,j,b;
    matrix X = make_matrix(50000, 3072);
    matrix y = make_matrix(50000, 10);
    d.X = X;
    d.y = y;


    for(b = 0; b < 5; ++b){
        char buff[256];
        sprintf(buff, "data/cifar/cifar-10-batches-bin/data_batch_%d.bin", b+1);
        FILE *fp = fopen(buff, "rb");
        if(!fp) file_error(buff);
        for(i = 0; i < 10000; ++i){
            unsigned char bytes[3073];
            fread(bytes, 1, 3073, fp);
            int class_id = bytes[0];
            y.vals[i+b*10000][class_id] = 1;
            for(j = 0; j < X.cols; ++j){
                X.vals[i+b*10000][j] = (double)bytes[j+1];
            }
        }
        fclose(fp);
    }
    //normalize_data_rows(d);
    //translate_data_rows(d, -128);
    scale_data_rows(d, 1./255);
    smooth_data(d);
    return d;
}

data load_go(char *filename)
{
    FILE *fp = fopen(filename, "rb");
    matrix X = make_matrix(3363059, 361);
    matrix y = make_matrix(3363059, 361);
    int row, col;

    if(!fp) file_error(filename);
    char *label;
    int count = 0;
    while((label = fgetl(fp))){
        int i;
        if(count == X.rows){
            X = resize_matrix(X, count*2);
            y = resize_matrix(y, count*2);
        }
        sscanf(label, "%d %d", &row, &col);
        char *board = fgetl(fp);

        int index = row*19 + col;
        y.vals[count][index] = 1;

        for(i = 0; i < 19*19; ++i){
            float val = 0;
            if(board[i] == '1') val = 1;
            else if(board[i] == '2') val = -1;
            X.vals[count][i] = val;
        }
        ++count;
        free(label);
        free(board);
    }
    X = resize_matrix(X, count);
    y = resize_matrix(y, count);

    data d = {0};
    d.shallow = 0;
    d.X = X;
    d.y = y;


    fclose(fp);

    return d;
}


void randomize_data(data d)
{
    int i;
    for(i = d.X.rows-1; i > 0; --i){
        int index = random_gen()%i;
        float *swap = d.X.vals[index];
        d.X.vals[index] = d.X.vals[i];
        d.X.vals[i] = swap;

        swap = d.y.vals[index];
        d.y.vals[index] = d.y.vals[i];
        d.y.vals[i] = swap;
    }
}

void scale_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        scale_array(d.X.vals[i], d.X.cols, s);
    }
}

void translate_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        translate_array(d.X.vals[i], d.X.cols, s);
    }
}

void normalize_data_rows(data d)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        normalize_array(d.X.vals[i], d.X.cols);
    }
}

data get_data_part(data d, int part, int total)
{
    data p = {0};
    p.shallow = 1;
    p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
    p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
    p.X.cols = d.X.cols;
    p.y.cols = d.y.cols;
    p.X.vals = d.X.vals + d.X.rows * part / total;
    p.y.vals = d.y.vals + d.y.rows * part / total;
    return p;
}

data get_random_data(data d, int num)
{
    data r = {0};
    r.shallow = 1;

    r.X.rows = num;
    r.y.rows = num;

    r.X.cols = d.X.cols;
    r.y.cols = d.y.cols;

    r.X.vals = calloc(num, sizeof(float *));
    r.y.vals = calloc(num, sizeof(float *));

    int i;
    for(i = 0; i < num; ++i){
        int index = random_gen()%d.X.rows;
        r.X.vals[i] = d.X.vals[index];
        r.y.vals[i] = d.y.vals[index];
    }
    return r;
}

data *split_data(data d, int part, int total)
{
    data *split = calloc(2, sizeof(data));
    int i;
    int start = part*d.X.rows/total;
    int end = (part+1)*d.X.rows/total;
    data train;
    data test;
    train.shallow = test.shallow = 1;

    test.X.rows = test.y.rows = end-start;
    train.X.rows = train.y.rows = d.X.rows - (end-start);
    train.X.cols = test.X.cols = d.X.cols;
    train.y.cols = test.y.cols = d.y.cols;

    train.X.vals = calloc(train.X.rows, sizeof(float*));
    test.X.vals = calloc(test.X.rows, sizeof(float*));
    train.y.vals = calloc(train.y.rows, sizeof(float*));
    test.y.vals = calloc(test.y.rows, sizeof(float*));

    for(i = 0; i < start; ++i){
        train.X.vals[i] = d.X.vals[i];
        train.y.vals[i] = d.y.vals[i];
    }
    for(i = start; i < end; ++i){
        test.X.vals[i-start] = d.X.vals[i];
        test.y.vals[i-start] = d.y.vals[i];
    }
    for(i = end; i < d.X.rows; ++i){
        train.X.vals[i-(end-start)] = d.X.vals[i];
        train.y.vals[i-(end-start)] = d.y.vals[i];
    }
    split[0] = train;
    split[1] = test;
    return split;
}

