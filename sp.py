import argparse
import json


class CWS:
    def __init__(self):
        self.values = dict()
        self.acc = dict()
        self.last_step = dict()
        self.step = 0

    def get_value(self, key, default):
        if key not in self.values:
            return default
        if self.last_step == None:
            return self.values[key]
        return self.get_update_value(key)

    def get_update_value(self, key):
        self.acc[key] += (self.step - self.last_step[key]) * (self.values[key])
        self.last_step[key] = self.step
        return self.values[key]

    def update_weights(self, key, delta):
        if key not in self.values:
            # 如果一开始不在 则初始化
            self.values[key] = 0
            self.acc[key] = 0
            self.last_step[key] = self.step
        else:
            # （如果需要的话）更新
            self.get_update_value(key)

        self.values[key] += delta

    def save(self, filename):
        # 保存训练好的模型
        json.dump({k: v for k, v in self.values.items() if v != 0.0},
                  open(filename, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)

    def load(self, filename):
        # 载入之前训练好的模型
        self.values.update(json.load(open(filename, encoding='utf-8')))
        self.last_step = None

    def gen_features(self, x):
        # 根据一个句子产生其每个字符对应的七个特征
        for i in range(len(x)):
            left2 = x[i - 2] if i - 2 >= 0 else '#'
            left1 = x[i - 1] if i - 1 >= 0 else '#'
            mid = x[i]
            right1 = x[i + 1] if i + 1 < len(x) else '#'
            right2 = x[i + 2] if i + 2 < len(x) else '#'
            features = ['1' + mid, '2' + left1, '3' + right1, '4' + left2 +
                        left1, '5' + left1 + mid, '6' + mid + right1, '7' + right1 + right2, '8' + left1 + mid + right1]
            yield features

    def update_cws(self, x, y, delta):
        for i, features in zip(range(len(x)), self.gen_features(x)):
            for feature in features:
                self.update_weights(str(y[i]) + feature, delta)
        for i in range(len(x) - 1):
            # 更新隐式马尔科夫中隐含层的转移概率
            self.update_weights(str(y[i]) + ':' + str(y[i + 1]), delta)

    def decode(self, x):  # 类似隐马模型的动态规划解码算法
        # 得到所有隐含层的转移概率
        transfer = []
        for i in range(4):
            transfer.append([self.get_value(str(i) + ':' + str(j), 0)
                             for j in range(4)])

        # 得到所有隐层的生成概率
        generate = []
        for features in self.gen_features(x):
            temp = []
            for tag in range(4):
                temp.append(sum(self.get_value(str(tag) + feature, 0)
                                for feature in features))
            generate.append(temp)

        # 计算前向概率
        # alphas是个三维数组 alphas[i][j][0]表示到第i个字符标记为j的概率 alphas[i][j][1]则相应地记录取到最大值的指针
        alphas = [[[e, None] for e in generate[0]]]
        for i in range(len(x) - 1):
            temp = []
            for k in range(4):
                # 统计最大概率和相应的指针
                temp.append(max([alphas[i][j][0] + transfer[j]
                                 [k] + generate[i + 1][k], j] for j in range(4)))
            alphas.append(temp)

        # 根据alphas中记录的指针倒推得到最优解序列
        alpha = max([alphas[-1][j], j] for j in range(4))
        i = len(x)
        tags = []
        while i:
            tags.append(alpha[1])
            i -= 1
            alpha = alphas[i][alpha[1]]
        return list(reversed(tags))


def seg_to_sent(words):
    # 根据一个已经分词的句子得到该句子的 x, y 向量
    # BEMS分别对应0123
    y = []
    for word in words:
        if len(word) == 1:
            y.append(3)
        else:
            y.extend([0] + [1] * (len(word) - 2) + [2])
    return ''.join(words), y


def sent_to_seg(x, y):
    # 根据 x, y 向量得到一个句子的分词结果
    cache = ''
    words = []
    for i in range(len(x)):
        cache += x[i]
        if y[i] == 2 or y[i] == 3:
            words.append(cache)
            cache = ''
    if cache:
        words.append(cache)
    return words


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', type=str, help='')
    parser.add_argument('--predict', type=str, help='')
    parser.add_argument('--result', type=str, help='')
    parser.add_argument('--iteration', type=int, default=5, help='')
    parser.add_argument('--load', type=str, help='')
    parser.add_argument('--save', type=str, help='')
    args = parser.parse_args()

    if args.train and not args.load:
        cws = CWS()
        for i in range(args.iteration):
            print('starting %d th train....' % (i + 1))
            line_cnt = 1
            for l in open(args.train, encoding='utf-8'):
                if line_cnt % 20000 == 0:
                    print('processing the %d th line' % (line_cnt))
                line_cnt += 1
                x, y = seg_to_sent(l.split())
                z = cws.decode(x)
                cws.step += 1
                if z != y:
                    cws.update_cws(x, y, 1)
                    cws.update_cws(x, z, -1)
            for key in cws.values:
                cws.get_update_value(key)
            for key in cws.acc:
                cws.values[key] = cws.acc[key] / cws.step

        if args.save:
            cws.save(args.save)
            print('successfully save the model into %s' % (args.save))

    if args.load:
        cws = CWS()
        cws.load(args.load)
        print('successfully load the model in %s' % (args.load))

    if args.predict and args.result:
        fin = open(args.predict, 'r', encoding='utf-8')
        fout = open(args.result, 'w', encoding='utf-8')
        for l in fin:
            l = l.strip()
            z = cws.decode(l)
            print(' '.join(sent_to_seg(l, z)), file=fout)
        fin.close()
        fout.close()
