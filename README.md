# KLMZYDA
Kullback–Leibler divergence based Music Zipper Yoking with Domain Adaptation

本工作旨在通过K首歌曲的片段拼接成为与目标歌曲风格相似的歌曲。

基本思路如下：
搜集目标音乐文件与歌词，根据歌词的断句切分音乐片段为Y={l1,l2,l3,...,ln}

搜集待拼接音乐文件与歌词每首歌的片段为N={h1,h2,h3,...,hmi}_{i=1,2,..,K}

目标是从集合N里拼接出一段音乐，要求：
1.拼接片段之间连贯。
2.相邻不能是一首歌。
3.拼成的歌曲与目标歌曲之间较为相似。
要完成如上目标我们需要明确
1）什么是相似，连贯。

2）可否总结为优化问题。

3）如何求解此优化问题。

针对1）我们认为，“连贯”是片段拼接部分信号频谱相似，“相似”是整体信号频谱相似。因此定义如下
连贯度(h,w)=KL(语谱图（h[最后2秒]）,语谱图（w[开头2秒]）)
相似度(h,w)=KL(语谱图（h[all]）,语谱图（w[all]）)

针对2），我们假设w'
min{sum_i连贯度(h_i,h_{i+1}) + sum_i连贯度(l_i,h_{i})}
针对3）此为非凸问题，采用模拟退火算法优化得到h_i

目标歌曲为李健的《向往》
五首备选歌曲与编号为
0-向往
1－贝加尔湖畔
2-传奇
3-风吹麦浪
4-什刹海
5－异乡人
