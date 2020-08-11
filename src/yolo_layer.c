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

// 构造YOLOV3的yolo层
// batch 一个batch中包含图片的张数
// w 输入图片的宽度
// h 输入图片的高度
// n 一个cell预测多少个bbox,这里是每个yolo层的cell预测box数目为3
// total total Anchor bbox的数目
// mask 使用的是0,1,2 还是
// classes 网络需要识别的物体类别数
layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes)
{
    int i;
    layer l = {0};
    l.type = YOLO;//层类别

    l.n = n;//一个cell预测多少个bbox
    l.total = total;//anchors的数目，为6
    l.batch = batch;// 一个batch包含图片的张数
    l.h = h;// 输入图片的宽度
    l.w = w;// 输入图片的高度
    l.c = n*(classes + 4 + 1);// 输入图片的通道数, 3*(20 + 5)
    l.out_w = l.w;// 输出图片的宽度
    l.out_h = l.h;// 输出图片的高度
    l.out_c = l.c;// 输出图片的通道数
    l.classes = classes;//目标类别数
    l.cost = calloc(1, sizeof(float));//yolo层总的损失
    l.biases = calloc(total*2, sizeof(float));//存储bbox的Anchor box的[w,h] total=6
    if(mask) l.mask = mask;//yolov3有mask传入
    else{
        l.mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
	//存储bbox的Anchor box的[w,h]的更新值
    l.bias_updates = calloc(n*2, sizeof(float));
	// 一张训练图片经过yolo层后得到的输出元素个数（等于网格数*每个网格预测的矩形框数*每个矩形框的参数个数）
#ifdef OPEN_OCC_CLASS_FLAG
    l.outputs = h*w*n*(classes + 4 + 1 + 1);
#else
    l.outputs = h*w*n*(classes + 4 + 1);
#endif
	//一张训练图片输入到yolo层的元素个数（注意是一张图片，对于yolo_layer，输入和输出的元素个数相等）
    l.inputs = l.outputs;
	
	//每张图片含有的真实矩形框参数的个数（max_boxes表示一张图片中最多有max_boxes个ground truth矩形框，每个真实矩形框有
	//5个参数，包括x,y,w,h四个定位参数，以及物体类别）,注意max_boxes是darknet程序内写死的，实际上每张图片可能
	//并没有max_boxes个真实矩形框，也能没有这么多参数，但为了保持一致性，还是会留着这么大的存储空间，只是其中的
	//值为空而已.
	l.max_boxes = max_boxes;
	// GT: max_boxes*(4+1) 存储max_boxes个bbox的信息，这里是假设图片中GT bbox的数量是
	//小于max_boxes的，这里是写死的；此处与yolov1是不同的
#ifdef OPEN_OCC_CLASS_FLAG
	l.truths = l.max_boxes*(4 + 1 + 1);    // 90*(4 + 1);
#else
	l.truths = l.max_boxes*(4 + 1);    // 90*(4 + 1);
#endif
   
	// yolo层误差项(包含整个batch的)
    l.delta = calloc(batch*l.outputs, sizeof(float));
    
	//yolo层所有输出（包含整个batch的）
	//yolo的输出维度是l.out_w*l.out_h，等于输出的维度，输出的通道数为l.out_c，也即是输入的通道数，具体为：n*(classes+coords+1)
	//YOLO检测模型将图片分成S*S个网格，每个网格又预测B个矩形框，最后输出的就是这些网格中包含的所有矩形框的信息

	l.output = calloc(batch*l.outputs, sizeof(float));
	// 存储bbox的Anchor box的[w,h]的初始化,在src/parse.c中parse_yolo函数会加载cfg中Anchor尺寸
	for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }
	// yolo层的前向传播
    l.forward = forward_yolo_layer;
	// yolo层的反向传播
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
//float *x：当前batch的图片输出参数起始地址
//float *biases：偏置项起始地址，预定义anchor box的储存位置
//int n = mask[n]：anchor box的大小索引
//int index:	boxIndex.在特征图中该box对应的索引
//i是宽度方向cell索引，j是高度方向cell索引，lw和lh分别为13 13，w h为网络的宽高416 416 stride为13*13
box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
	////biases 存储bbox的Anchor box的[w,h]
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
//函数作用：得到需要进行预测激活的值（特征图中的点）的特征图中的位置
//batch用于选择哪个图像的信息
//n用于选择哪个bbox的信息，每个box都是13*13*（1+4+classes）的大小，矩阵可表示为（4+1+classes）*（13*13）
//entry用于指定哪一类的信息 0 对应的就是内存中，预测的x y w h 信息，即：box_index
//loc是列的意思，即：第多少个cell
//https://blog.csdn.net/haithink/article/details/94006918
//entry_index(l, 0, n*l.w*l.h + i, 4);
static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);//n对应每个cell的第n个预测 box（共mask = 3个预测box）
    int loc = location % (l.w*l.h);//loc表示定位到每个feature map中 行列 cell的具体位置
	//	  （batch中图片偏移量）+（图片中box的偏移量）+（box中对不同变量的偏移量）+ （当前特征图中的cell位置）
    //return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h +         loc;
#ifdef OPEN_OCC_CLASS_FLAG
	return batch*l.outputs + n*l.w*l.h*(4 + l.classes + 1 + 1) + entry*l.w*l.h + loc;//将feature map中的每个prebox的位置，对应到内存中返回
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
	//将网络的输入复制到层的输出中。
	memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));

#ifndef GPU
	for (b = 0; b < l.batch; ++b) 
	{
		for (n = 0; n < l.n; ++n) //遍历对应的cell预测出的n个anchor
		{
			/*
			在输入大小为13*13的yolo层中,假设类别数为80,
			每个batch中一张图像对应的数据个数为 3*13*13*(4+1+80)
			这些数据在内存中是连续的

			13*13==169个cell, 每个cell产生3个bounding box，每个bounding box 数据为 13*13*(4+1+80
			————————————————
			版权声明：本文为CSDN博主「haithink」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
			原文链接：https://blog.csdn.net/haithink/article/details/94006918
			
			
			*/
			//获得 box 的 index
			// 即通过该cell对应的anchors与truth的iou来判断使用哪一个anchor产生的predict来回归
			int index = entry_index(l, b, n*l.w*l.h, 0);
			// 获得 box 的预测 x , y , w , h,注意都是相对值,不是真实坐标
			//（1.0/(1.0+exp(-x))），激活x,y  // 对 tx, ty进行logistic变换

			//why 2?  2就是表示只对(x,y)进行了激活,有没有想到什么,对!
			//就是论文中的那个bx = sigmoid(tx)+cx,
			//				  by = sigmoid(ty)+cy,
			//作者就是在这里对tx,ty做sigmoid处理的

			activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);//掉2 * 13 *13个数进行sigmoid逻辑回归
			index = entry_index(l, b, n*l.w*l.h, 4);//从特征图开始，跳过x,y,h,w直接定位到confidence和Id类标签的预测

			// 对confidence和C类进行logistic变换
			activate_array(l.output + index, (1 + l.classes)*l.w*l.h, LOGISTIC);
#ifdef OPEN_OCC_CLASS_FLAG
			//找到遮挡的索引
			index = entry_index(l, b, n*l.w*l.h, 4 +l.classes + 1);//找到anchor box最后一个地址的遮挡变量的位置
			//进行逻辑回归
			activate_array(l.output + index, 1 * l.w*l.h, LOGISTIC);//对每个anchor box的 1 * 13 *13个数（判断遮挡还是不遮挡）进行sigmoid逻辑回归
#endif
		}
	}
#endif
	// 初始化梯度
	memset(l.delta, 0, l.outputs * l.batch * sizeof(float));//将l.delta中的元素清零。每次前向传播前都进行了清零工作哦
	if (!state.train) return;
	float avg_iou = 0;
	float recall = 0;
	float recall75 = 0;
	float avg_cat = 0;
	float avg_head_occ = 0;//记录没有遮挡
	float avg_tail = 0;//记录有遮挡

	float uncertain_Count = 0;//记录有遮挡
	float avg_uncertain = 0;//记录不确定类别

	float avg_obj = 0;
	float avg_anyobj = 0;
	int count = 0;
	int head_count = 0;
	int tail_Count = 0;
	int class_count = 0;
	*(l.cost) = 0;//初始化损失为0
	
	// 下面四个for循环是依次取n个预测的box的 x，y, w,h,confidence,class，然后依次和所有groud true 计算IOU，取IOU最大的groud true.
	for (b = 0; b < l.batch; ++b) //iter for one batch images
	{
		for (j = 0; j < l.h; ++j) //col grid cell
		{
			for (i = 0; i < l.w; ++i) //row grid cell
			{
				for (n = 0; n < l.n; ++n) //pre box
				{
					//遍历每张图片中的j行i列，第n个pre box            
					// 对每个预测的bounding box
					// 找到与其IoU最大的ground truth
					int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);// 取pre box,在索引中，每个box的位置算一个数，传入到第二个参数中
					//根据上一步计算的box所在内存地址，把predict box 的信息取出来
					box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.w*l.h);
					float best_iou = 0;// 最大IOU
					int best_t = 0;// 和pre对用最大IOU的groud truth的索引
					for (t = 0; t < l.max_boxes; ++t) // 计算 最大IOU及其索引
					{//和一张图片中所有的GT做IOU比较，只取一个IOU最高的匹配。
						/*
						state.truth中保存ground truth 在一个batch2张图片中
						图片1[ [x y w h id] [x y w h id] [x y w h id] [x y w h id] [x y w h id] ... ]默认一共能存90个box，每个box5个变量
						图片2[ [x y w h id] [x y w h id] [x y w h id] [x y w h id] [x y w h id] ... ]
							
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
					int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);//读取预测的objectness索引
					avg_anyobj += l.output[obj_index]; //读取预测的objectness值,avg_anyobj训练状态检测量

					// 计算 confidence的偏差 如果如果iou小于0.7就认为是负样本了，负样本的损失就累计到了delta中
					l.delta[obj_index] = 0 - l.output[obj_index];//默认作为负样本,目标objectness=0,误差设置为0 - l.output[obj_index]
					// 大于IOU设置的阈值 confidence梯度设为0
					if (best_iou > l.ignore_thresh) 
					{
						l.delta[obj_index] = 0;//若大于指定的阈值，则不计算类别损失//iou较大,不能作为负样本, 清除误差
					}
					// yolov3这段代码不会执行，因为 l.truth_thresh值为1
					if (best_iou > l.truth_thresh) 
					{//这个参数在cfg文件中，值为1，这个条件语句永远不可能成立
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
		//for (b = 0; b < l.batch; ++b) //iter for one batch images
		//上面遍历了整张图片的每个cell中的每个pre box，下面代码处理的基本单元是整张图片（即，第n张图片）。
		//遍历该图片中的所有GT
		//// box,class 的梯度，只计算groud truth对应的预测框的梯： 先计算groud truth和所有anchoiou，然后选最大IOU的索引，若这个索引在mask里，计算梯度和loss.
			
		/*-------------------
		上述代码段，对每一张图片都计算了 3 个anchor box和GT阈值小于0.7的误差到delta中。换言之，除了很确定的正样本，其他的都进行了误差记录
		下述代码段，对所有的GT 和3个anchor box都计算IOU，找到iou最大的那个anchor的索引，如果找到了就看是不是哪一个yolo层的anchor，是的话才进行梯度计算和反向传播。只有最大iou的那个才反向传播
		-------------------*/
		//针对当前图片的处理
		for (t = 0; t < l.max_boxes; ++t) //遍历当前图片的所有GT框
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
				i = (truth.x * l.w);//GT的中心点位于第（i，j）个cell，也就是该cell负责预测这个truth。 // pre对应中心坐标y
				j = (truth.y * l.h);// pred 对应中心坐标x
				box truth_shift = truth;
				truth_shift.x = truth_shift.y = 0;//将truth_shift的box位置移动到0,0 
				for (n = 0; n < l.total; ++n) // 和所有anchor计算IOU   // 遍历每一个anchor bbox找到与GT bbox最大的IOU
				{//遍历每个先验框，找出与GT具有最大iou的先验框
					box pred = { 0 };
					pred.w = l.biases[2 * n] / state.net.w;//net.w表示图片的大小 // 计算pred bbox的w在相对整张输入图片的位置
					pred.h = l.biases[2 * n + 1] / state.net.h;// 计算pred bbox的h在相对整张输入图片的位置
					float iou = box_iou(pred, truth_shift);//此iou是不考虑x,y，仅考虑w,h的得到的 // 计算GT box truth_shift 与 预测bbox pred二者之间的IOU
					if (iou > best_iou) {
						best_iou = iou;// 记录最大的IOU
						best_n = n;// 以及记录该bbox的编号n
					}
				}
				// 上面记录bbox的编号,是否由该层Anchor预测的
				// 判断最好AIOU对应的索引best_n 是否在mask里面，若没有，返回-1
				int mask_n = int_index(l.mask, best_n, l.n);//在l.mask数组指针中寻找best_n，若找到则返回best_n在l.mask中的下标，若找不到返回-1。
				//以下代码是针对正例
				if (mask_n >= 0)
				{
					int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
					//计算定位损失
					//以每张图的GT作为基准。先找到与GT有最大IOU的pre box，然后计算其产生的损失。有可能这个pre box产生的损失已经计算过了，又重新计算了一遍。
					// 计算box梯度  // 对box位置参数进行求导
					float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h);

					int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
					avg_obj += l.output[obj_index];

					// 计算梯度，公式(6)，梯度前面要加个”-“号， 1代表是真实标签
					l.delta[obj_index] = 1 - l.output[obj_index];
#ifdef OPEN_OCC_CLASS_FLAG
					//新增的遮挡属性 属性用sigmoid进行计算
					//1 获取变量偏移地址
					int occlude_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + l.classes + 1);
					//2 获取ground truth 遮挡属性：0为不遮挡，1为遮挡
					int class_occluda = state.truth[t*(4 + 1 + 1) + b*l.truths + 4 + 1];
					//3 计算梯度（偏差）采用sigmoid几乎函数
					//解析：如果标签是0 不遮挡 那么期望l.output的预测值为0 这样，0-0=0趋向于损失减少

					if (class_occluda < 3)//只有遮挡与否的ID为0或者1才进行遮挡逻辑回归，否则不计算损失
					{
						//如果打开遮挡训练，就损失回传
						//l.delta[occlude_index] = ((0 == class_occluda) ? 0 : 1) - l.output[occlude_index];
						//l.delta[occlude_index] = ((0 == class_occluda) ? 0 : 1) - l.output[occlude_index];
						if (0 == class_occluda)//车头类 0
						{
							l.delta[occlude_index] = 0 - l.output[occlude_index];
						}
						else if (1 == class_occluda)//车尾类1
						{
							l.delta[occlude_index] = 1 - l.output[occlude_index];
						}
						else//不确定类2
						{
							l.delta[occlude_index] = 0.5 - l.output[occlude_index];
						}


						if (class_occluda==0)//车头类别
						{
							head_count++;
							avg_head_occ += l.output[occlude_index];
						}
						else if (class_occluda == 1)//车尾类别
						{
							tail_Count++;
							avg_tail += l.output[occlude_index];
						}
						else//不确定类别 2
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
					// 获得best_iou对应anchor box的index
					int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);

					// 计算类别的梯度   // 对class进行求导
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
	//直接把 l.delta 拷贝给上一层的 delta。注意 net.delta 指向 prev_layer.delta。
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

