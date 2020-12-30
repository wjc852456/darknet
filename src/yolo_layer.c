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

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)  //完成 yolo 层初始化操作
{
    int i;
    layer l = {0};  //定义一个层
    l.type = YOLO;  // 类型设为YOLO

    l.n = n;  //该层每个grid预测的框的个数，yolov3.cfg 为3
    l.total = total;  //总anchors的对数，yolov3.cfg 为9
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1);  //输入和输出相等，yolo 层是最后一层，不需要把输出传递到下一层。每个 grid 预测 n 个 box，每个box预测 x,y,w,h,置信度和80种类别
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));  //误差代价分配空间
    l.biases = calloc(total*2, sizeof(float));  //保存 anchor 的大小，保存的是10,14,  23,27,  37,58这些
    if(mask) l.mask = mask;  //l.mask 里保存了 [yolo] 配置里 “mask = 0,1,2” 的数值
    else{
        l.mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;  //input和output数目相等
    l.truths = 90*(4 + 1);  //每张图片最多保存90个标签
    l.delta = calloc(batch*l.outputs, sizeof(float));  //反向传播误差导数
    l.output = calloc(batch*l.outputs, sizeof(float));  //预测结果
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;  //如果未指定 anchors，默认设置为0.5，否则在 ./src/parser.c 里会把 l.biases 的值设为 anchors 的大小
    }

    l.forward = forward_yolo_layer;  //函数指针，前向和后向计算
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

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)  //获得预测的边界框
{  //x就是predictions/output；biases就是anchors；n是anchor数量，这里是3；index是二维格点(batch,h,w)的位置；i,j是col,row；stride等于l.w*l.h
    box b; 
    b.x = (i + x[index + 0*stride]) / lw;  //预测中心 x 值在当前层里的相对位置
    b.y = (j + x[index + 1*stride]) / lh;  //预测中心 y 值在当前层里的相对位置
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;  //exp(x)*anchor/w，相对于网络输入 net.w 的宽度
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;  //exp(x)*anchor/h，相对于网络输入 net.h 的高度
    return b;
}

float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)  //计算预测边界框的误差
{
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);  //获得 index 位置预测的边界框
    float iou = box_iou(pred, truth);  //计算预测框与 label 的iou

    float tx = (truth.x*lw - i);  //与预测值相匹配
    float ty = (truth.y*lh - j);
    float tw = log(truth.w*w / biases[2*n]);  //log 使大框和小框的误差影响接近
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);  //计算误差 delta，乘了权重系数 scale=(2-truth.w*truth.h)
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}


void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat)  //计算类别误差
{
    int n;
    if (delta[index]){  //应该不会进入这个判断，因为 delta[index] 初值为0
        delta[index + stride*class] = 1 - output[index + stride*class];
        if(avg_cat) *avg_cat += output[index + stride*class];
        return;
    }
    for(n = 0; n < classes; ++n){  //对所有类别，如果预测正确，则误差为 1-predict，否则为 0-predict
        delta[index + stride*n] = ((n == class)?1 : 0) - output[index + stride*n];
        if(n == class && avg_cat) *avg_cat += output[index + stride*n];
    }
}

static int entry_index(layer l, int batch, int location, int entry)  //得到指针偏移量，即入口需要的索引
{
    int n =   location / (l.w*l.h);  //第几个框，每个 grid 有3个框
    int loc = location % (l.w*l.h);  //第几个 grid
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;  //返回第 loc 个 grid 的第 n 个框的 entry 的指针偏移位置
}

void forward_yolo_layer(const layer l, network net)  //最重要的前向传播
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));  //将层输入直接拷贝到层输出

#ifndef GPU  //在 cpu 里，把预测输出的 x,y,confident 和80种类别都 sigmoid 激活，确保值在0~1
    for (b = 0; b < l.batch; ++b){  //batch循环，确定当前是第几个batch
        for(n = 0; n < l.n; ++n){  //anchor循环，确定当前是第几个anchor
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC); //每个grid的x,y做sigmoid激活
            index = entry_index(l, b, n*l.w*l.h, 4);  //跳过x,y,w,h
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);  //每个grid的confident和80种类别做激活
        }
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));  //把 l.delta 初始化为0
    if(!net.train) return;  //非训练阶段则直接给出预测输出，以下是训练部分
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;  //损失值初始化
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {  //对该层每一个预测输出的框，共有 l.batch*l.h*l.w*l.n 个
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);  //获得 box 的起始位置，即 box.x 的位置
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h);  //获得该位置的 box 保存到 pred
                    float best_iou = 0;
                    int best_t = 0;
                    for(t = 0; t < l.max_boxes; ++t){  //找到与 pred 的 iou 最大的 label(ground truth)
                        box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);  //获得 pred 框的 confident 的位置
                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = 0 - l.output[obj_index];  //confident 误差
                    if (best_iou > l.ignore_thresh) {  //如果最大 iou 大于 ignore_thresh，则不计算损失
                        l.delta[obj_index] = 0;
                    }
                    if (best_iou > l.truth_thresh) {  //best_iou 不可能大于 truth_thresh(1)，所以不会进入
                        l.delta[obj_index] = 1 - l.output[obj_index];

                        int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class = l.map[class];
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0);
                        box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                    }
                }
            }
        }
        for(t = 0; t < l.max_boxes; ++t){  //对每一个 label(ground truth)
            box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);  //获得一个 label

            if(!truth.x) break;  //如果值为空，即没有更多标签，则结束
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);  //label 在层中的位置
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;  //不考虑 x,y，只考虑 w,h，即只考虑 iou
            for(n = 0; n < l.total; ++n){  //找到与 label 有最大 iou 的 anchor
                box pred = {0};
                pred.w = l.biases[2*n]/net.w;
                pred.h = l.biases[2*n+1]/net.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }

            int mask_n = int_index(l.mask, best_n, l.n);  //这个函数判断上面找到的 anchor 是否是该层要预测的
            if(mask_n >= 0){  //如果该 anchor 是该层要预测的
                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);  //获得该 anchor 在层输出中对应的 box 位置
                float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);  //计算 box 与 label 的误差

                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);  //获得该 box 的 confident 的位置
                avg_obj += l.output[obj_index];
                l.delta[obj_index] = 1 - l.output[obj_index];  //该位置应该有正样本，所以误差为 1-predict

                int class = net.truth[t*(4 + 1) + b*l.truths + 4];  //获得 label 的真实类别
                if (l.map) class = l.map[class];
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);  //获得 box 类别的起始位置0（80种类别）
                delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat);  //计算类别误差

                ++count;
                ++class_count;
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    //计算损失函数，cost=sum(l.delta*l.delta)
    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
}

void backward_yolo_layer(const layer l, network net)  //误差反向传播
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);  //直接把 l.delta 拷贝给上一层的 delta。注意 net.delta 指向 prev_layer.delta
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)  //调整预测 box 中心和大小
{  ////w 和 h 是输入图片的尺寸，netw 和 neth 是网络输入尺寸
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {  //新图片尺寸
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){  //调整 box 相对新图片尺寸的位置
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

int yolo_num_detections(layer l, float thresh)  //预测输出中置信度超过阈值的 box 个数
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);  //获得置信度偏移位置
            if(l.output[obj_index] > thresh){  //置信度超过阈值
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

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)  //获得预测输出中超过阈值的 box
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
            float objectness = predictions[obj_index];  //置信度
            if(objectness <= thresh) continue;  //置信度小于阈值，无目标，舍弃
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j){  //遍历所有的classes
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];  //类别置信度（置信度*类别概率）
                dets[count].prob[j] = (prob > thresh) ? prob : 0;  //小于阈值则概率置0
            }
            ++count;
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);  //调整 box 大小
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_yolo_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_yolo_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

