import argparse
import json


class CWS:
    def __init__(self):
        self.values = dict()
        self.acc = dict()
        self.last_step = dict()
        self.step = 0

    def get_value(self, key, default=0):
        if key in self.values:
            return self.values[key]
        return default

    def save_model(self, filename):
        json.dump({k: v for k, v in self.values.items()}, open(
            filename, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def load_model(self, filename):
        self.values.update(json.load(open(filename, 'r', encoding='utf-8')))

    def gen_features(self, x):
        # 根据一个句子产生其每个字符对应的七个特征
        for i in range(len(x)):
            left2 = x[i - 2] if i >= 2 else '#'
            left1 = x[i - 1] if i >= 1 else '#'
            mid = x[i]
            right1 = x[i + 1] if i + 1 < len(x) else '#'
            right2 = x[i + 2] if i + 2 < len(x) else '#'
            features = ['1' + mid, '2' + left1, '3' + right1, '4' + left2 +
                        left1, '5' + left1 + mid, '6' + mid + right1, '7' + right1 + right2, '8' + left1 + mid + right1]
            yield features

    def predict(self, x):
        # 感知器预测
        t_list = []
        for i, features in zip(range(len(x)), self.gen_features(x)):
            temp = 0
            for feature in features:
                temp += self.get_value(feature)
            # 如果是0默认认为要分词
            temp = 1 if temp >= 0 else -1
            t_list.append(temp)
        return t_list


def seg_to_sent(words):
    # 根据一个已经分词的句子得到该句子的 x, y 向量
    # 1是和前面一个字分开 -1不分开
    y = []
    for word in words:
        if len(word) == 1:
            y.append(1)
        else:
            y.extend([1] + [-1] * (len(word) - 1))
    return ''.join(words), y


def sent_to_seg(x, y):
    # 根据 x, y 向量得到一个句子的分词结果
    cache = ''
    words = []
    for i in range(len(x)):
        if y[i] == 1 and cache != '':
            words.append(cache)
            cache = ''
        cache += x[i]
    words.append(cache)
    return words


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', type=str, help='')
    parser.add_argument('--predict', type=str, help='')
    parser.add_argument('--result', type=str, help='')
    parser.add_argument('--save', type=str, help='')
    parser.add_argument('--load', type=str, help='')
    parser.add_argument('--iteration', type=int, default=5, help='')
    args = parser.parse_args()

    if args.train and not args.load:
        cws = CWS()
        for i in range(args.iteration):
            print('starting %d th train....' % (i + 1))
            line_cnt = 0
            fin = open(args.train, 'r', encoding='utf-8')
            for l in fin:
                x, y = seg_to_sent(l.split())
                z = cws.predict(x)
                if z != y:
                    # update, y is correct, z is predict
                    for i, features in zip(range(len(y)), cws.gen_features(x)):
                        for feature in features:
                            if feature in cws.values:
                                # old value accumulated
                                cws.acc[feature] += (cws.step -
                                                     cws.last_step[feature]) * cws.values[feature]
                                cws.acc[feature] += y[i] - z[i]
                                cws.values[feature] += y[i] - z[i]
                                cws.last_step[feature] = cws.step + 1
                            else:
                                cws.values[feature] = y[i] - z[i]
                                cws.acc[feature] = y[i] - z[i]
                                cws.last_step[feature] = cws.step + 1
                # update step count
                cws.step += 1

                line_cnt += 1
                if line_cnt % 10000 == 0:
                    print(line_cnt)
        # update acc
        for key in cws.acc:
        	if cws.last_step[key] != cws.step:
        		cws.acc[key] += (cws.step - cws.last_step[key]) * cws.values[key]

        for key in cws.acc:
            cws.acc[key] /= cws.step
            # cws.values[key] = cws.acc[key] / cws.step

        if args.save:
            # save the model
            cws.save_model(args.save)
            print('successfully save the model into %s' % (args.save))

        print(cws.step)

    if args.load:
        # load the model
        cws = CWS()
        cws.load_model(args.load)
        print('successfully load the model in %s' % (args.load))

    if args.predict and args.result:
        fin = open(args.predict, 'r', encoding='utf-8')
        fout = open(args.result, 'w', encoding='utf-8')
        for l in fin:
            l = l.strip()
            z = cws.predict(l)
            print(' '.join(sent_to_seg(l, z)), file=fout)
        fin.close()
        fout.close()
