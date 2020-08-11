#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

// ����YOLOV3��yolo��
// batch һ��batch�а���ͼƬ������
// w ����ͼƬ�Ŀ��
// h ����ͼƬ�ĸ߶�
// n һ��cellԤ����ٸ�bbox,������ÿ��yolo���cellԤ��box��ĿΪ3
// total total Anchor bbox����Ŀ
// mask ʹ�õ���0,1,2 ����
// classes ������Ҫʶ������������
layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes)
{
    int i;
    layer l = {0};
    l.type = YOLO;//�����

    l.n = n;//һ��cellԤ����ٸ�bbox
    l.total = total;//anchors����Ŀ��Ϊ6
    l.batch = batch;// һ��batch����ͼƬ������
    l.h = h;// ����ͼƬ�Ŀ��
    l.w = w;// ����ͼƬ�ĸ߶�
    l.c = n*(classes + 4 + 1);// ����ͼƬ��ͨ����, 3*(20 + 5)
    l.out_w = l.w;// ���ͼƬ�Ŀ��
    l.out_h = l.h;// ���ͼƬ�ĸ߶�
    l.out_c = l.c;// ���ͼƬ��ͨ����
    l.classes = classes;//Ŀ�������
    l.cost = calloc(1, sizeof(float));//yolo���ܵ���ʧ
    l.biases = calloc(total*2, sizeof(float));//�洢bbox��Anchor box��[w,h] total=6
    if(mask) l.mask = mask;//yolov3��mask����
    else{
        l.mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
	//�洢bbox��Anchor box��[w,h]�ĸ���ֵ
    l.bias_updates = calloc(n*2, sizeof(float));
	// һ��ѵ��ͼƬ����yolo���õ������Ԫ�ظ���������������*ÿ������Ԥ��ľ��ο���*ÿ�����ο�Ĳ���������
#ifdef OPEN_OCC_CLASS_FLAG
    l.outputs = h*w*n*(classes + 4 + 1 + 1);
#else
    l.outputs = h*w*n*(classes + 4 + 1);
#endif
	//һ��ѵ��ͼƬ���뵽yolo���Ԫ�ظ�����ע����һ��ͼƬ������yolo_layer������������Ԫ�ظ�����ȣ�
    l.inputs = l.outputs;
	
	//ÿ��ͼƬ���е���ʵ���ο�����ĸ�����max_boxes��ʾһ��ͼƬ�������max_boxes��ground truth���ο�ÿ����ʵ���ο���
	//5������������x,y,w,h�ĸ���λ�������Լ��������,ע��max_boxes��darknet������д���ģ�ʵ����ÿ��ͼƬ����
	//��û��max_boxes����ʵ���ο�Ҳ��û����ô���������Ϊ�˱���һ���ԣ����ǻ�������ô��Ĵ洢�ռ䣬ֻ�����е�
	//ֵΪ�ն���.
	l.max_boxes = max_boxes;
	// GT: max_boxes*(4+1) �洢max_boxes��bbox����Ϣ�������Ǽ���ͼƬ��GT bbox��������
	//С��max_boxes�ģ�������д���ģ��˴���yolov1�ǲ�ͬ��
#ifdef OPEN_OCC_CLASS_FLAG
	l.truths = l.max_boxes*(4 + 1 + 1);    // 90*(4 + 1);
#else
	l.truths = l.max_boxes*(4 + 1);    // 90*(4 + 1);
#endif
   
	// yolo�������(��������batch��)
    l.delta = calloc(batch*l.outputs, sizeof(float));
    
	//yolo�������������������batch�ģ�
	//yolo�����ά����l.out_w*l.out_h�����������ά�ȣ������ͨ����Ϊl.out_c��Ҳ���������ͨ����������Ϊ��n*(classes+coords+1)
	//YOLO���ģ�ͽ�ͼƬ�ֳ�S*S������ÿ��������Ԥ��B�����ο��������ľ�����Щ�����а��������о��ο����Ϣ

	l.output = calloc(batch*l.outputs, sizeof(float));
	// �洢bbox��Anchor box��[w,h]�ĳ�ʼ��,��src/parse.c��parse_yolo���������cfg��Anchor�ߴ�
	for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }
	// yolo���ǰ�򴫲�
    l.forward = forward_yolo_layer;
	// yolo��ķ��򴫲�
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "yolo\n");
    srand(0);

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
#ifdef OPEN_OCC_CLASS_FLAG
	l->outputs = h*w*l->n*(l->classes + 4 + 1 + 1);
#else
	l->outputs = h*w*l->n*(l->classes + 4 + 1);
#endif
   
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}
//float *x����ǰbatch��ͼƬ���������ʼ��ַ
//float *biases��ƫ������ʼ��ַ��Ԥ����anchor box�Ĵ���λ��
//int n = mask[n]��anchor box�Ĵ�С����
//int index:	boxIndex.������ͼ�и�box��Ӧ������
//i�ǿ�ȷ���cell������j�Ǹ߶ȷ���cell������lw��lh�ֱ�Ϊ13 13��w hΪ����Ŀ��416 416 strideΪ13*13
box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
	////biases �洢bbox��Anchor box��[w,h]
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}


void delta_yolo_class(float *output, float *delta, int index, int class_id, int classes, int stride, float *avg_cat, int focal_loss)
{
    int n;
    if (delta[index + stride*class_id]){
        delta[index + stride*class_id] = 1 - output[index + stride*class_id];
        if(avg_cat) *avg_cat += output[index + stride*class_id];
        return;
    }
    // Focal loss
    if (focal_loss) {
        // Focal Loss
        float alpha = 0.5;    // 0.25 or 0.5
        //float gamma = 2;    // hardcoded in many places of the grad-formula

        int ti = index + stride*class_id;
        float pt = output[ti] + 0.000000000000001F;
        // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
        float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);    // http://blog.csdn.net/linmingan/article/details/77885832
        //float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);    // https://github.com/unsky/focal-loss

        for (n = 0; n < classes; ++n) {
            delta[index + stride*n] = (((n == class_id) ? 1 : 0) - output[index + stride*n]);

            delta[index + stride*n] *= alpha*grad;

            if (n == class_id) *avg_cat += output[index + stride*n];
        }
    }
    else {
        // default
        for (n = 0; n < classes; ++n) {
            delta[index + stride*n] = ((n == class_id) ? 1 : 0) - output[index + stride*n];
			if (n == class_id && avg_cat) {
				*avg_cat += output[index + stride*n];
			}
				
        }
    }
}
//�������ã��õ���Ҫ����Ԥ�⼤���ֵ������ͼ�еĵ㣩������ͼ�е�λ��
//batch����ѡ���ĸ�ͼ�����Ϣ
//n����ѡ���ĸ�bbox����Ϣ��ÿ��box����13*13*��1+4+classes���Ĵ�С������ɱ�ʾΪ��4+1+classes��*��13*13��
//entry����ָ����һ�����Ϣ 0 ��Ӧ�ľ����ڴ��У�Ԥ���x y w h ��Ϣ������box_index
//loc���е���˼�������ڶ��ٸ�cell
//https://blog.csdn.net/haithink/article/details/94006918
//entry_index(l, 0, n*l.w*l.h + i, 4);
static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);//n��Ӧÿ��cell�ĵ�n��Ԥ�� box����mask = 3��Ԥ��box��
    int loc = location % (l.w*l.h);//loc��ʾ��λ��ÿ��feature map�� ���� cell�ľ���λ��
	//	  ��batch��ͼƬƫ������+��ͼƬ��box��ƫ������+��box�жԲ�ͬ������ƫ������+ ����ǰ����ͼ�е�cellλ�ã�
    //return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h +         loc;
#ifdef OPEN_OCC_CLASS_FLAG
	return batch*l.outputs + n*l.w*l.h*(4 + l.classes + 1 + 1) + entry*l.w*l.h + loc;//��feature map�е�ÿ��prebox��λ�ã���Ӧ���ڴ��з���
#else
	return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
#endif
}

static box float_to_box_stride(float *f, int stride)
{//x y w h id occFlag
    box b = { 0 };
    b.x = f[0];
    b.y = f[1 * stride];
    b.w = f[2 * stride];
    b.h = f[3 * stride];
    return b;
}

void forward_yolo_layer(const layer l, network_state state)
{
	int i, j, b, t, n;
	//����������븴�Ƶ��������С�
	memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));

#ifndef GPU
	for (b = 0; b < l.batch; ++b) 
	{
		for (n = 0; n < l.n; ++n) //������Ӧ��cellԤ�����n��anchor
		{
			/*
			�������СΪ13*13��yolo����,���������Ϊ80,
			ÿ��batch��һ��ͼ���Ӧ�����ݸ���Ϊ 3*13*13*(4+1+80)
			��Щ�������ڴ�����������

			13*13==169��cell, ÿ��cell����3��bounding box��ÿ��bounding box ����Ϊ 13*13*(4+1+80
			��������������������������������
			��Ȩ����������ΪCSDN������haithink����ԭ�����£���ѭ CC 4.0 BY-SA ��ȨЭ�飬ת���븽��ԭ�ĳ������Ӽ���������
			ԭ�����ӣ�https://blog.csdn.net/haithink/article/details/94006918
			
			
			*/
			//��� box �� index
			// ��ͨ����cell��Ӧ��anchors��truth��iou���ж�ʹ����һ��anchor������predict���ع�
			int index = entry_index(l, b, n*l.w*l.h, 0);
			// ��� box ��Ԥ�� x , y , w , h,ע�ⶼ�����ֵ,������ʵ����
			//��1.0/(1.0+exp(-x))��������x,y  // �� tx, ty����logistic�任

			//why 2?  2���Ǳ�ʾֻ��(x,y)�����˼���,��û���뵽ʲô,��!
			//���������е��Ǹ�bx = sigmoid(tx)+cx,
			//				  by = sigmoid(ty)+cy,
			//���߾����������tx,ty��sigmoid�����

			activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);//��2 * 13 *13��������sigmoid�߼��ع�
			index = entry_index(l, b, n*l.w*l.h, 4);//������ͼ��ʼ������x,y,h,wֱ�Ӷ�λ��confidence��Id���ǩ��Ԥ��

			// ��confidence��C�����logistic�任
			activate_array(l.output + index, (1 + l.classes)*l.w*l.h, LOGISTIC);
#ifdef OPEN_OCC_CLASS_FLAG
			//�ҵ��ڵ�������
			index = entry_index(l, b, n*l.w*l.h, 4 +l.classes + 1);//�ҵ�anchor box���һ����ַ���ڵ�������λ��
			//�����߼��ع�
			activate_array(l.output + index, 1 * l.w*l.h, LOGISTIC);//��ÿ��anchor box�� 1 * 13 *13�������ж��ڵ����ǲ��ڵ�������sigmoid�߼��ع�
#endif
		}
	}
#endif
	// ��ʼ���ݶ�
	memset(l.delta, 0, l.outputs * l.batch * sizeof(float));//��l.delta�е�Ԫ�����㡣ÿ��ǰ�򴫲�ǰ�����������㹤��Ŷ
	if (!state.train) return;
	float avg_iou = 0;
	float recall = 0;
	float recall75 = 0;
	float avg_cat = 0;
	float avg_head_occ = 0;//��¼û���ڵ�
	float avg_tail = 0;//��¼���ڵ�

	float uncertain_Count = 0;//��¼���ڵ�
	float avg_uncertain = 0;//��¼��ȷ�����

	float avg_obj = 0;
	float avg_anyobj = 0;
	int count = 0;
	int head_count = 0;
	int tail_Count = 0;
	int class_count = 0;
	*(l.cost) = 0;//��ʼ����ʧΪ0
	
	// �����ĸ�forѭ��������ȡn��Ԥ���box�� x��y, w,h,confidence,class��Ȼ�����κ�����groud true ����IOU��ȡIOU����groud true.
	for (b = 0; b < l.batch; ++b) //iter for one batch images
	{
		for (j = 0; j < l.h; ++j) //col grid cell
		{
			for (i = 0; i < l.w; ++i) //row grid cell
			{
				for (n = 0; n < l.n; ++n) //pre box
				{
					//����ÿ��ͼƬ�е�j��i�У���n��pre box            
					// ��ÿ��Ԥ���bounding box
					// �ҵ�����IoU����ground truth
					int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);// ȡpre box,�������У�ÿ��box��λ����һ���������뵽�ڶ���������
					//������һ�������box�����ڴ��ַ����predict box ����Ϣȡ����
					box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.w*l.h);
					float best_iou = 0;// ���IOU
					int best_t = 0;// ��pre�������IOU��groud truth������
					for (t = 0; t < l.max_boxes; ++t) // ���� ���IOU��������
					{//��һ��ͼƬ�����е�GT��IOU�Ƚϣ�ֻȡһ��IOU��ߵ�ƥ�䡣
						/*
						state.truth�б���ground truth ��һ��batch2��ͼƬ��
						ͼƬ1[ [x y w h id] [x y w h id] [x y w h id] [x y w h id] [x y w h id] ... ]Ĭ��һ���ܴ�90��box��ÿ��box5������
						ͼƬ2[ [x y w h id] [x y w h id] [x y w h id] [x y w h id] [x y w h id] ... ]
							
						*/
#ifdef OPEN_OCC_CLASS_FLAG
						box truth = float_to_box_stride(state.truth + t*(4 + 1 + 1) + b*l.truths, 1);
						int class_id = state.truth[t*(4 + 1 + 1) + b*l.truths + 4];
#else
						box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);
						int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
#endif
						if (class_id >= l.classes) 
						{
							printf(" Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] \n", class_id, l.classes, l.classes - 1);
							getchar();
							continue; // if label contains class_id more than number of classes in the cfg-file
						}
						if (!truth.x) break;  // continue;
						float iou = box_iou(pred, truth);
						if (iou > best_iou) 
						{
							best_iou = iou;
							best_t = t;
						}
					}
					int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);//��ȡԤ���objectness����
					avg_anyobj += l.output[obj_index]; //��ȡԤ���objectnessֵ,avg_anyobjѵ��״̬�����

					// ���� confidence��ƫ�� ������iouС��0.7����Ϊ�Ǹ������ˣ�����������ʧ���ۼƵ���delta��
					l.delta[obj_index] = 0 - l.output[obj_index];//Ĭ����Ϊ������,Ŀ��objectness=0,�������Ϊ0 - l.output[obj_index]
					// ����IOU���õ���ֵ confidence�ݶ���Ϊ0
					if (best_iou > l.ignore_thresh) 
					{
						l.delta[obj_index] = 0;//������ָ������ֵ���򲻼��������ʧ//iou�ϴ�,������Ϊ������, ������
					}
					// yolov3��δ��벻��ִ�У���Ϊ l.truth_threshֵΪ1
					if (best_iou > l.truth_thresh) 
					{//���������cfg�ļ��У�ֵΪ1��������������Զ�����ܳ���
					//������YOLOv3�������еĵ��Ľ��ᵽ���ⲿ�֡�
					//���߳���Faster R-CNN���ᵽ��˫IoU���ԣ���anchor��GT��IoU����0.7ʱ����anchor������������������ʧ�С�
					//��ѵ�������в�û�в����õ�Ч���������������ˡ�
						l.delta[obj_index] = 1 - l.output[obj_index];//����Ŀ��Ŀ�����Խ����delta[obj_index]ԽС

						int class_id = state.truth[best_t*(4 + 1) + b*l.truths + 4];//�õ�GT�������(0,1,2,3,4)�ֱ��ʾ��x,y,w,h,class����������4
						if (l.map) class_id = l.map[class_id]; //���ͳһת����ֱ�Ӻ���
						int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);//��λ��������

						//���������ʧ
						delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, 0, l.focal_loss);
						box truth = float_to_box_stride(state.truth + best_t*(4 + 1) + b*l.truths, 1);
						//���㶨λ��ʧ
						//��ÿ��Ԥ���Ϊ��׼����ÿ��cell��Ӧ��Ԥ���ȥ���GT����IOU������ֵ���������ʧ����ע�����һ��delta_yolo_box������Ŷ����
						//��������ֵ���ƣ������п�������и����GTû��ƥ�䵽��Ӧ��Ԥ���©���ⲿ�ֵ���ʧ��
						//����ֻ��Ԥ�����GT֮���IOU�����趨����ֵ��������λ��ʧ
						delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h);
					}
				}
			}
		}
		//for (b = 0; b < l.batch; ++b) //iter for one batch images
		//�������������ͼƬ��ÿ��cell�е�ÿ��pre box��������봦��Ļ�����Ԫ������ͼƬ��������n��ͼƬ����
		//������ͼƬ�е�����GT
		//// box,class ���ݶȣ�ֻ����groud truth��Ӧ��Ԥ�����ݣ� �ȼ���groud truth������anchoiou��Ȼ��ѡ���IOU�������������������mask������ݶȺ�loss.
			
		/*-------------------
		��������Σ���ÿһ��ͼƬ�������� 3 ��anchor box��GT��ֵС��0.7����delta�С�����֮�����˺�ȷ�����������������Ķ�����������¼
		��������Σ������е�GT ��3��anchor box������IOU���ҵ�iou�����Ǹ�anchor������������ҵ��˾Ϳ��ǲ�����һ��yolo���anchor���ǵĻ��Ž����ݶȼ���ͷ��򴫲���ֻ�����iou���Ǹ��ŷ��򴫲�
		-------------------*/
		//��Ե�ǰͼƬ�Ĵ���
		for (t = 0; t < l.max_boxes; ++t) //������ǰͼƬ������GT��
			{
#ifdef OPEN_OCC_CLASS_FLAG
				box truth = float_to_box_stride(state.truth + t*(4 + 1 + 1) + b*l.truths, 1);
				int class_id = state.truth[t*(4 + 1 + 1) + b*l.truths + 4];
#else
				box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);
				int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
#endif
				if (class_id >= l.classes) continue; // if label contains class_id more than number of classes in the cfg-file

				if (!truth.x) break;  // continue;
				float best_iou = 0;
				int best_n = 0;
				i = (truth.x * l.w);//GT�����ĵ�λ�ڵڣ�i��j����cell��Ҳ���Ǹ�cell����Ԥ�����truth�� // pre��Ӧ��������y
				j = (truth.y * l.h);// pred ��Ӧ��������x
				box truth_shift = truth;
				truth_shift.x = truth_shift.y = 0;//��truth_shift��boxλ���ƶ���0,0 
				for (n = 0; n < l.total; ++n) // ������anchor����IOU   // ����ÿһ��anchor bbox�ҵ���GT bbox����IOU
				{//����ÿ��������ҳ���GT�������iou�������
					box pred = { 0 };
					pred.w = l.biases[2 * n] / state.net.w;//net.w��ʾͼƬ�Ĵ�С // ����pred bbox��w�������������ͼƬ��λ��
					pred.h = l.biases[2 * n + 1] / state.net.h;// ����pred bbox��h�������������ͼƬ��λ��
					float iou = box_iou(pred, truth_shift);//��iou�ǲ�����x,y��������w,h�ĵõ��� // ����GT box truth_shift �� Ԥ��bbox pred����֮���IOU
					if (iou > best_iou) {
						best_iou = iou;// ��¼����IOU
						best_n = n;// �Լ���¼��bbox�ı��n
					}
				}
				// �����¼bbox�ı��,�Ƿ��ɸò�AnchorԤ���
				// �ж����AIOU��Ӧ������best_n �Ƿ���mask���棬��û�У�����-1
				int mask_n = int_index(l.mask, best_n, l.n);//��l.mask����ָ����Ѱ��best_n�����ҵ��򷵻�best_n��l.mask�е��±꣬���Ҳ�������-1��
				//���´������������
				if (mask_n >= 0)
				{
					int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
					//���㶨λ��ʧ
					//��ÿ��ͼ��GT��Ϊ��׼�����ҵ���GT�����IOU��pre box��Ȼ��������������ʧ���п������pre box��������ʧ�Ѿ�������ˣ������¼�����һ�顣
					// ����box�ݶ�  // ��boxλ�ò���������
					float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h);

					int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
					avg_obj += l.output[obj_index];

					// �����ݶȣ���ʽ(6)���ݶ�ǰ��Ҫ�Ӹ���-���ţ� 1��������ʵ��ǩ
					l.delta[obj_index] = 1 - l.output[obj_index];
#ifdef OPEN_OCC_CLASS_FLAG
					//�������ڵ����� ������sigmoid���м���
					//1 ��ȡ����ƫ�Ƶ�ַ
					int occlude_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + l.classes + 1);
					//2 ��ȡground truth �ڵ����ԣ�0Ϊ���ڵ���1Ϊ�ڵ�
					int class_occluda = state.truth[t*(4 + 1 + 1) + b*l.truths + 4 + 1];
					//3 �����ݶȣ�ƫ�����sigmoid��������
					//�����������ǩ��0 ���ڵ� ��ô����l.output��Ԥ��ֵΪ0 ������0-0=0��������ʧ����

					if (class_occluda < 3)//ֻ���ڵ�����IDΪ0����1�Ž����ڵ��߼��ع飬���򲻼�����ʧ
					{
						//������ڵ�ѵ��������ʧ�ش�
						//l.delta[occlude_index] = ((0 == class_occluda) ? 0 : 1) - l.output[occlude_index];
						//l.delta[occlude_index] = ((0 == class_occluda) ? 0 : 1) - l.output[occlude_index];
						if (0 == class_occluda)//��ͷ�� 0
						{
							l.delta[occlude_index] = 0 - l.output[occlude_index];
						}
						else if (1 == class_occluda)//��β��1
						{
							l.delta[occlude_index] = 1 - l.output[occlude_index];
						}
						else//��ȷ����2
						{
							l.delta[occlude_index] = 0.5 - l.output[occlude_index];
						}


						if (class_occluda==0)//��ͷ���
						{
							head_count++;
							avg_head_occ += l.output[occlude_index];
						}
						else if (class_occluda == 1)//��β���
						{
							tail_Count++;
							avg_tail += l.output[occlude_index];
						}
						else//��ȷ����� 2
						{
							uncertain_Count++;
							avg_uncertain += l.output[occlude_index];
						}
					}
					else
					{
						l.delta[occlude_index] = 0;
					}

					int class_id = state.truth[t*(4 + 1 +1) + b*l.truths + 4];
#else
					int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
#endif
					if (l.map) class_id = l.map[class_id];
					// ���best_iou��Ӧanchor box��index
					int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);

					// ���������ݶ�   // ��class������
					delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, &avg_cat, l.focal_loss);

					++count;
					++class_count;
					if (iou > .5) recall += 1;
					if (iou > .75) recall75 += 1;
					avg_iou += iou;
				}
			}
	}
		//���洦����������batch�е�����ͼƬ
		*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);//��delta�е�ֵת��Ϊһ������ֵ�洢��l.lossָ��ĵ�ַ�С�
		//mag_array(float *a, int n)�Ĺ����ǣ���������ָ���и���ƽ���ĺͣ��ٿ�����
#ifdef OPEN_OCC_CLASS_FLAG
		//if (head_count == 0)	head_count = 1;
		//if (tail_Count == 0)	tail_Count = 1;
		//if (uncertain_Count == 0)	uncertain_Count = 1;

		printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f, \n Head: %f, Tail: %f, Uncertain: %f, count: %d\n", state.index, avg_iou / count, 
			avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, avg_head_occ / head_count, avg_tail / tail_Count, avg_uncertain/ uncertain_Count, count);
#else
		printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", state.index, avg_iou / count, avg_cat / class_count, 
			avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count);
#endif
}

void backward_yolo_layer(const layer l, network_state state)
{
	//ֱ�Ӱ� l.delta ��������һ��� delta��ע�� net.delta ָ�� prev_layer.delta��
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (letter) {
        if (((float)netw / w) < ((float)neth / h)) {
            new_w = netw;
            new_h = (h * netw) / w;
        }
        else {
            new_h = neth;
            new_w = (w * neth) / h;
        }
    }
    else {
        new_w = netw;
        new_h = neth;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}
/*
��������ÿ��batch�е�һ��ͼƬ�����ذ��������Ԥ���ĸ���
*/
int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter)
{
    int i,j,n;
    float *predictions = l.output;
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
#ifdef OPEN_OCC_CLASS_FLAG
			int occ_index = entry_index(l, 0, n*l.w*l.h + i, 4 + l.classes + 1);
			float occness= predictions[occ_index];
#endif
            if(objectness <= thresh) continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
#ifdef OPEN_OCC_CLASS_FLAG
			dets[count].occness = occness;
#endif
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
				//printf("\n prob = %f", prob);
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network_state state)
{
    copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_ongpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_ongpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
#ifdef OPEN_OCC_CLASS_FLAG
			//-----------------------add by qielizhong---------------------
			index = entry_index(l, b, n*l.w*l.h, 4 + l.classes + 1);
			activate_array_ongpu(l.output_gpu + index, 1*l.w*l.h, LOGISTIC);
			//-------------------------------------------------------------
#endif
        }
    }
    if(!state.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    //cuda_pull_array(l.output_gpu, state.input, l.batch*l.inputs);
    float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
    cuda_pull_array(l.output_gpu, in_cpu, l.batch*l.inputs);
    float *truth_cpu = 0;
    if (state.truth) {
        int num_truth = l.batch*l.truths;
        truth_cpu = calloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    network_state cpu_state = state;
    cpu_state.net = state.net;
    cpu_state.index = state.index;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_yolo_layer(l, cpu_state);
    //forward_yolo_layer(l, state);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    free(in_cpu);
    if (cpu_state.truth) free(cpu_state.truth);
}

void backward_yolo_layer_gpu(const layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
}
#endif

