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

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes)
{
    int i;
    layer l = {0};
    l.type = YOLO;

    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(total*2, sizeof(float));
    if(mask) l.mask = mask;
    else{
        l.mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    l.max_boxes = max_boxes;
    l.truths = l.max_boxes*(4 + 1);    // 90*(4 + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
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

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
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

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
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
            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
        }
    }
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);//n对应每个cell的第n个pre box
    int loc = location % (l.w*l.h);//loc表示定位到cell的具体位置
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

static box float_to_box_stride(float *f, int stride)
{
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
	//将网络的输入复制到层的输出中。
	memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));

#ifndef GPU
	for (b = 0; b < l.batch; ++b) 
	{
		for (n = 0; n < l.n; ++n) 
		{
			int index = entry_index(l, b, n*l.w*l.h, 0);
			//（1.0/(1.0+exp(-x))），激活x,y
			activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);
			index = entry_index(l, b, n*l.w*l.h, 4);
			//用logistic激活(c,C1,C2,C3...)
			activate_array(l.output + index, (1 + l.classes)*l.w*l.h, LOGISTIC);
		}
	}
#endif

	memset(l.delta, 0, l.outputs * l.batch * sizeof(float));//将l.delta中的元素清零。每次前向传播前都进行了清零工作哦
	if (!state.train) return;
	float avg_iou = 0;
	float recall = 0;
	float recall75 = 0;
	float avg_cat = 0;
	float avg_obj = 0;
	float avg_anyobj = 0;
	int count = 0;
	int class_count = 0;
	*(l.cost) = 0;//初始化损失为0
	for (b = 0; b < l.batch; ++b) {
		for (j = 0; j < l.h; ++j) {
			for (i = 0; i < l.w; ++i) {
				for (n = 0; n < l.n; ++n) {
					//遍历每张图片中的j行i列，第n个pre box
						int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
						box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.w*l.h);
						float best_iou = 0;
						int best_t = 0;
						for (t = 0; t < l.max_boxes; ++t) {//和一张图片中所有的GT做IOU比较，只取一个IOU最高的匹配。
							box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);
							int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
							if (class_id >= l.classes) {
								printf(" Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] \n", class_id, l.classes, l.classes - 1);
								getchar();
								continue; // if label contains class_id more than number of classes in the cfg-file
							}
							if (!truth.x) break;  // continue;
							float iou = box_iou(pred, truth);
							if (iou > best_iou) {
								best_iou = iou;
								best_t = t;
							}
						}
						int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
						avg_anyobj += l.output[obj_index];

						l.delta[obj_index] = 0 - l.output[obj_index];//delta取一个负值？正负无所谓的，因为最后计算(*l.loss)时都要平方。

						if (best_iou > l.ignore_thresh) {
							l.delta[obj_index] = 0;//若大于指定的阈值，则不计算类别损失
						}
						if (best_iou > l.truth_thresh) {//这个参数在cfg文件中，值为1，这个条件语句永远不可能成立
														//作者在YOLOv3的论文中的第四节提到了这部分。
														//作者尝试Faster R-CNN中提到的双IoU策略，当anchor与GT的IoU大于0.7时，该anchor被算作正样本计入损失中。
														//但训练过程中并没有产生好的效果，所以最后放弃了。
							l.delta[obj_index] = 1 - l.output[obj_index];//包含目标的可能性越大，则delta[obj_index]越小

							int class_id = state.truth[best_t*(4 + 1) + b*l.truths + 4];//得到GT所属类别。(0,1,2,3,4)分别表示（x,y,w,h,class），这里是4
							if (l.map) class_id = l.map[class_id]; //类别统一转换，直接忽略
							int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);//定位到类别概率

							//计算类别损失
							delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, 0, l.focal_loss);
							box truth = float_to_box_stride(state.truth + best_t*(4 + 1) + b*l.truths, 1);
							//计算定位损失
							//以每个预测框为基准。让每个cell对应的预测框去拟合GT，若IOU大于阈值，则计算损失。（注意和另一个delta_yolo_box的区别哦！）
							//由于有阈值限制，这样有可能造成有个别的GT没有匹配到对应的预测框，漏了这部分的损失。
							//这样只把预测框与GT之间的IOU大于设定的阈值的算作定位损失
							delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h);
						}
					}
				}
			}
			//上面遍历了整张图片的每个cell中的每个pre box，下面代码处理的基本单元是整张图片（即，第n张图片）。
			//遍历该图片中的所有GT
			for (t = 0; t < l.max_boxes; ++t) {
				box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);
				int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
				if (class_id >= l.classes) continue; // if label contains class_id more than number of classes in the cfg-file

				if (!truth.x) break;  // continue;
				float best_iou = 0;
				int best_n = 0;
				i = (truth.x * l.w);//GT的中心点位于第（i，j）个cell，也就是该cell负责预测这个truth。
				j = (truth.y * l.h);
				box truth_shift = truth;
				truth_shift.x = truth_shift.y = 0;
				for (n = 0; n < l.total; ++n) {//遍历每个先验框，找出与GT具有最大iou的先验框
					box pred = { 0 };
					pred.w = l.biases[2 * n] / state.net.w;//net.w表示图片的大小
					pred.h = l.biases[2 * n + 1] / state.net.h;
					float iou = box_iou(pred, truth_shift);//此iou是不考虑x,y，仅考虑w,h的得到的
					if (iou > best_iou) {
						best_iou = iou;
						best_n = n;
					}
				}

				int mask_n = int_index(l.mask, best_n, l.n);//在l.mask数组指针中寻找best_n，若找到则返回best_n在l.mask中的下标，若找不到返回-1。
				if (mask_n >= 0) {
					int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
					//计算定位损失
					//以每张图的GT作为基准。先找到与GT有最大IOU的pre box，然后计算其产生的损失。有可能这个pre box产生的损失已经计算过了，又重新计算了一遍。
					float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h);

					int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
					avg_obj += l.output[obj_index];
					l.delta[obj_index] = 1 - l.output[obj_index];

					int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
					if (l.map) class_id = l.map[class_id];
					int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
					delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, &avg_cat, l.focal_loss);

					++count;
					++class_count;
					if (iou > .5) recall += 1;
					if (iou > .75) recall75 += 1;
					avg_iou += iou;
				}
			}
		}
		//上面处理完了整个batch中的所有图片
		*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);//将delta中的值转化为一个浮点值存储到l.loss指向的地址中。
		//mag_array(float *a, int n)的功能是：先求数组指针中各项平方的和，再开方。
		printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", state.index, avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count);
	}
}
void backward_yolo_layer(const layer l, network_state state)
{
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
检测器检测每个batch中第一张图片，返回包含物体的预测框的个数
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
            if(objectness <= thresh) continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
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

